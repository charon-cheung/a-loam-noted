// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk
#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;

const int systemDelay = 0; 
int systemInitCount = 0;
bool systemInited = false;
int N_SCANS = 0;
float cloudCurvature[400000];
int cloudSortInd[400000];
int cloudNeighborPicked[400000];
int cloudLabel[400000];

bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }

ros::Publisher pubLaserCloud;
//大曲率特征点——角点
ros::Publisher pubCornerPointsSharp;
//小曲率特征点——降采样角点
ros::Publisher pubCornerPointsLessSharp;
//平面点
ros::Publisher pubSurfPointsFlat;
//降采样平面点
ros::Publisher pubSurfPointsLessFlat;
// 剔除点
ros::Publisher pubRemovePoints;
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;

double MINIMUM_RANGE = 0.1; 


// 这个函数是模仿 pcl::removeNaNFromPointCloud 函数写的
template <typename PointT>
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    // pcl库的常见操作，判断入参和出参是否相同
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;

    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + 
            cloud_in.points[i].y * cloud_in.points[i].y + 
            cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    // 去掉多余点
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    // width：类型为uint32_t，表示点云宽度（如果组织为图像结构），即一行点云的数量。

    /*  height：类型为uint32_t，表示点云高度（如果组织为图像结构）
    若为有序点云，height 可以大于 1，即多行点云，每行固定点云的宽度
    若为无序点云，height 等于1，即一行点云，此时 width 的数量即为点云的数量  */

    // is_dense：bool 类型，若点云中的数据都是有限的（不包含 inf/NaN 这样的值），则为 true，否则为 false
    /*   ROS收到的是有序点云，高度为线数，宽度为每个scan的宽度
    但是经过处理之后，不再是原来的特征，不能保证每个线上都有点   */

    cloud_out.height = 1;
    // 宽度为总数
    cloud_out.width = static_cast<uint32_t>(j);
    cloud_out.is_dense = true;
}

void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (!systemInited)
    {
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    TicToc t_whole;
    TicToc t_prepare;
    std::vector<int> scanStartInd(N_SCANS, 0);
    std::vector<int> scanEndInd(N_SCANS, 0);

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
    std::vector<int> indices;
    // 输入点云不变
    // 输出电云为去除NaN点后的无序点云
    // index为无序点云在输入点云中的索引
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);


    int cloudSize = laserCloudIn.points.size();
    // 激光雷达为顺时针旋转，而角度逆时针才为正
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;
    // 保证激光扫描的范围在 pi ~ 3pi
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    bool halfPassed = false;
    int count = cloudSize;
    PointType point;
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);

    // 这里的计算方式其实并不适用kitti数据集的点云，因为kitti的点云已经被运动补偿过
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
        // 计算点云垂直俯仰角
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;  // 激光帧的序号

        // 现在Lidar的驱动通常给出了点在第几根线，LIO_SAM里直接就用了，没有自己计算
        // 计算它是第几根线
        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);
        // 求水平角，获得点在线的位置
        float ori = -atan2(point.y, point.x);
        // 判断是否扫描一半，进行补偿，保证扫描范围在pi-2pi
        if (!halfPassed)
        {
            // 确保 PI /2 < ori - startOri < 3/2 * PI
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            // 这种情况不会发生
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }
            // 超过起始角PI
            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        // 当前点到起始点的百分比，分母一般为2π
        float relTime = (ori - startOri) / (endOri - startOri);
        /* 整数部分+小数部分：整数是scan的索引
        小数部分是相对时间（旋转了多少，对应的时间）
        scanPeriod写死为0.1，也就是一圈0.1s，小数部分不超过0.1,完成了按照时间排序 */

        // 目前的雷达驱动中基本已经有了，不必自己处理
        // 强度值实际没有用到，intensity是保存自己需要的量，后面没有用到
        point.intensity = scanID + scanPeriod * relTime;
        // 把数据分散插入每条线，每条线一个数组，后面的提取特征是在每条线分别进行
        laserCloudScans[scanID].push_back(point); 
    }
    cloudSize = count;  // 更新点云数量，有效点会变少
    printf("points size %d \n", cloudSize);


    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    // std::vector<int> scanStartInd(N_SCANS, 0);
    // std::vector<int> scanEndInd(N_SCANS, 0);
    // 全部集合到一个点云里，但使用两个数组标记起始和结果
    // 计算曲率时，最左和最右的5个点不计算曲率，少10个点影响很小
    // 开始和结束处的点云容易产生不闭合的“接缝”，对提取edge feature不利 
    for (int i = 0; i < N_SCANS; i++)
    {
        scanStartInd[i] = laserCloud->size() + 5;
        ROS_INFO("scanStartInd %d : %d", i, scanStartInd[i] );
        /*  将上面的子点云laserCloudScans合并成一帧点云 laserCloud
        这里的单帧点云laserCloud可以认为已经是有序点云了，按照scanID和fireID的顺序存放*/
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
        ROS_INFO("scanEndInd %d : %d", i, scanEndInd[i] );
    }
    printf("prepare time %f \n", t_prepare.toc());
    /* 预处理部分结束 */


    /*    特征提取部分   */
    /*  角点曲率大，面点曲率小。计算曲率，除了开始5个点和结束的5个点
    以当前点为中心，前5个和后5个点都参与计算  */
    for (int i = 5; i < cloudSize - 5; i++)
    {
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        // 存储曲率，索引
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;
        // 两个标志位
        // 点有没有被选为feature点，如果之后选为特征点之后，将被设置为1
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;   // 曲率的分类

    }
    TicToc t_pts;

    pcl::PointCloud<PointType> cornerPointsSharp;
    pcl::PointCloud<PointType> cornerPointsLessSharp;
    pcl::PointCloud<PointType> surfPointsFlat;
    pcl::PointCloud<PointType> surfPointsLessFlat;

    float t_q_sort = 0;
    // 整个for循环是根据曲率计算 4 种特征点
    for (int i = 0; i < N_SCANS; i++)
    {
        // 如果该scan的点数少于7个点，就跳过
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);

        // 将该scan分成6小段执行特征检测
        // 由于int精度的问题这里可能并不是严格6等分的
        for (int j = 0; j < 6; j++)     
        {
            //起始索引，六等分的起点
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6;
            //结束索引，-1避免重复， 六等分的终点
            // 若不减一，下一份的sp与上一份的ep处于相同位置，这样可能会重复提取同一个特征点
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            // 根据曲率，从小到大对subscan的点进行sort
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);
            // 对排序也进行计时
            t_q_sort += t_tmp.toc();


            // 从后往前，即从曲率大的点开始提取corner feature
            int largestPickedNum = 0;
            for (int k = ep; k >= sp; k--)
            {
                // 排序之后，索引就乱了，但我们保存了原来的索引
                int ind = cloudSortInd[k]; 
                // 如果该点没有被选择过，并且曲率大于0.1
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {
                    largestPickedNum++;
                    // 该subscan中曲率最大的前2个点当做corner_sharp特征点
                    if (largestPickedNum <= 2)
                    {                        
                        cloudLabel[ind] = 2;
                        cornerPointsSharp.push_back(laserCloud->points[ind]);
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    // 曲率第3到第20，不再是corner_sharp，而是corner_less_sharp
                    else if (largestPickedNum <= 20)
                    {
                        cloudLabel[ind] = 1;
                        // 该subscan中曲率最大的前20个点认为是corner_less_sharp特征点
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    } 
                    // 不能提取太多，计算量大；太少会使精度差
                    else
                    {
                        break;
                    }
                    // 标记该点被选择过了，不一定有20个点
                    cloudNeighborPicked[ind] = 1; 
                    //将曲率比较大的点的前五个距离比较近的点筛选出去，防止特征点的聚集
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        // 如果相邻点的距离差异过大，则点云不连续或者是特征边缘，就是新的特征
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)  // 写死
                        {
                            break;
                        }
                        // 将选中的点周围5个点都置为1，避免后续选到
                        // 认为是比较近的点，标记为1,也就是选择处理过
                        cloudNeighborPicked[ind + l] = 1;
                    }
                    // 这里是后5个点
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }
                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            // 提取曲率比较小的点（平面点）
            // 因为曲率从小到大排序了，所以从小索引开始寻找
            int smallestPickedNum = 0;
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];
                // 如果曲率比较小，并且没有被筛选过
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {
                    //  -1表示曲率很小的点  surf_flat
                    cloudLabel[ind] = -1; 
                    surfPointsFlat.push_back(laserCloud->points[ind]);
                    // 只选取曲率最小的前四个点
                    // 不区分平坦和比较平坦的点
                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)
                    { 
                        break;
                    }
                    //  该点被选择处理过了，标志设为1
                    cloudNeighborPicked[ind] = 1;
                    // 跟角点的处理一样，防止特征点聚集，将距离比较近的点筛选出去
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }
            //将当前这条scan剩余的点(包括surf_flat)，组成surf_less_flat进行降采样
            for (int k = sp; k <= ep; k++)
            {   // 除了角点之外的所有点，实际中角点很少
                if (cloudLabel[k] <= 0)
                {
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }

        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        // 体素滤波
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);
        // 新变量赋值
        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);
    printf("seperate points time %f \n", t_pts.toc());

    // 发布总的点云
    sensor_msgs::PointCloud2 laserCloudOutMsg;
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
    laserCloudOutMsg.header.frame_id = "/camera_init";
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "/camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "/camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // pub each scan
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "/camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    printf("scan registration time %f ms \n", t_whole.toc());
    // 时间太长会丢帧
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms !");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");
    ros::NodeHandle nh;

    nh.param<int>("scan_line", N_SCANS, 16);
    // remove too closed points to lidar
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);

    printf("scan line number %d \n", N_SCANS);

    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }

    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points", 100, laserCloudHandler);

    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);

    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);

    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);

    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);

    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);

    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);

    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();

    return 0;
}
