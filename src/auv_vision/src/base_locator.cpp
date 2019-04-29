#include "ros/ros.h"
#include "geometry_msgs/Pose.h"
#include <sstream>
#include <util/ImgprocUtil.h>
#include <dynamic_reconfigure/server.h>
#include "common/AbstractImageConverter.h"
#include "auv_common/Base.h"
#include "auv_common/Euler.h"
#include <tf/transform_datatypes.h>
#include <nav_msgs/Odometry.h>
#include <tf/tf.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


#include "beacons/BeaconDetector.h"
#include "beacons/BeaconDescriptor.h"
#include "markers/ArUcoDetector.h"

#define BEACONS_NUMBER 9
#define LINES_NUMBER 36 /// 0.5*BEACONS_NUMBER*(BEACONS_NUMBER-1)
#define PARALLEL_CRITERIA 1
#define LENGTH_CRITERIA 2

static const std::string OPENCV_WINDOW = "Image window";
static const std::string OPENCV_WINDOW_BEACONS_NUMBERS = "Beacons Numbers";

static const std::string ENABLE_WINDOWS_PARAM = "debugVision";

static const std::string CAMERA_BOTTOM_TOPIC = "/cam_bottom/image_raw";
static const std::string CAMERA_FRONT_TOPIC = "/cam_front_1/image_raw";

static const std::string BASE_PUBLISH_TOPIC_BEACONS = "/base/beacons";
static const std::string BASE_PUBLISH_TOPIC_MARKERS = "/base/markers";
static const std::string BASE_PUBLISH_TOPIC_BEACONS_Q = "/base/beacons/quaternions";
static const std::string BASE_PUBLISH_TOPIC_MARKERS_Q = "/base/markers/quaternions";
static const std::string BASE_PUBLISH_TOPIC_MARKERS_ROS = "/base/ros_markers";
static const std::string BASE_PUBLISH_TOPIC_ODOM = "/base/odom/angle";

//static const std::string ARUCO_PUBLISH_TOPIC = "/base/ros_markers";

static const std::string BASE_LOCATOR_NODE_NAME = "base_locator";

BeaconDetector beaconDetector;
ArUcoDetector ArUcoDetector;

bool station_detected = false;

cv::Mat DoubleMatFromVec3b(cv::Vec3b in)
{
    cv::Mat mat(3,1, CV_64FC1);
    mat.at <double>(0,0) = in [0];
    mat.at <double>(1,0) = in [1];
    mat.at <double>(2,0) = in [2];

    return mat;
};

void getEulerAngles(cv::Mat &rotCamerMatrix, cv::Vec3d &eulerAngles) {

    cv::Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
    double* _r = rotCamerMatrix.ptr<double>();
    double projMatrix[12] = {_r[0],_r[1],_r[2],0,
                             _r[3],_r[4],_r[5],0,
                             _r[6],_r[7],_r[8],0};

    cv::decomposeProjectionMatrix( cv::Mat(3,4,CV_64FC1,projMatrix), /// Decomposes a projection matrix into a rotation matrix and a camera matrix
                                   cameraMatrix,
                                   rotMatrix,
                                   transVect,
                                   rotMatrixX,
                                   rotMatrixY,
                                   rotMatrixZ,
                                   eulerAngles);
}

cv::Vec4f getLineFromPoints(const cv::Point2f& point_1, const cv::Point2f& point_2) {
    cv::Vec4f line;
    line[0] = point_1.x;
    line[1] = point_1.y;
    line[2] = point_2.x;
    line[3] = point_2.y;
    return line;
}

/// Sorting function (sorts elements from small to big in one array's row) (Insertion Sort Method)
void sorting(std::vector<std::vector<float>>& angle, int row, int low_ind, int high_ind) { /// Row is a number of sorting row
    int i, j;
    float key_0, key_1, key_2;
    for (i = low_ind + 1; i <= high_ind; i++)
    {
        //std::cout<<angle[1][i]<<" - Angle"<<std::endl;
        key_0 = angle[0][i];
        key_1 = angle[1][i];
        key_2 = angle[2][i];

        j = i-1;

        /** Move elements of arr[0..i-1], that are
        greater than key, to one position ahead
        of their current position */
        if (row == 1) {
            while (j >= 0 && angle[1][j] > key_1)
            {
                angle[0][j+1] = angle[0][j];
                angle[1][j+1] = angle[1][j];
                angle[2][j+1] = angle[2][j];

                j = j-1;
            }
        }

        if (row == 2) {
            while (j >= 0 && angle[2][j] > key_2)
            {
                angle[0][j+1] = angle[0][j];
                angle[1][j+1] = angle[1][j];
                angle[2][j+1] = angle[2][j];

                j = j-1;
            }
        }
        angle[0][j+1] = key_0;
        angle[1][j+1] = key_1;
        angle[2][j+1] = key_2;
    }
}

/// Sorting function (sorts elements from small to big in one array's row) (Insertion Sort Method)
void point_sorting(std::vector<cv::Point2f>& point, int size, bool sort_y)
{
    float x, y;
    int i, j;
    for (i = 1; i < size; i++) {
        x = point[i].x;
        y = point[i].y;
        j = i - 1;

        /** Move elements of arr[0..i-1], that are
          greater than key, to one position ahead
          of their current position */
        if (sort_y) {
            while (j >= 0 && point[j].y > y) {
                point[j + 1].x = point[j].x;
                point[j + 1].y = point[j].y;
                j = j - 1;
            }
        }
        else {
            while (j >= 0 && point[j].x > x) {
                point[j + 1].x = point[j].x;
                point[j + 1].y = point[j].y;
                j = j - 1;
            }
        }
        point[j + 1].x = x;
        point[j + 1].y = y;
    }
}

float getLineSlope(const cv::Vec4f &line) {
    if (line[0] == line[2])
        return 90;
    float tan = std::abs(line[1] - line[3]) / std::abs(line[0] - line[2]);
    float arctan = std::atan(tan);
    float res = arctan * 180.0f / CV_PI;
    return res;
}

float getDistance(float x1, float y1, float x2, float y2) {
    return std::sqrt((x1 - x2)*(x1 - x2) - (y1 - y2)*(y1 - y2));
}

float getLength(const cv::Vec4f &line) {
    return std::sqrt((line[2]-line[0])*(line[2]-line[0]) + (line[3]-line[1])*(line[3]-line[1]));
}

std::vector<cv::Point3f> Generate3DPoints()
{
    std::vector<cv::Point3f> points;

    float x,y,z;

    x=-1083;y=1578;z=500;
    points.push_back(cv::Point3f(x,y,z));

    x=-1083;y=1578;z=-500;
    points.push_back(cv::Point3f(x,y,z));

    x=0;y=1578;z=260;
    points.push_back(cv::Point3f(x,y,z));

    x=0;y=1578;z=-260;
    points.push_back(cv::Point3f(x,y,z));

    x=260;y=1578;z=0;
    points.push_back(cv::Point3f(x,y,z));

    x=1083;y=1578;z=500;
    points.push_back(cv::Point3f(x,y,z));

    x=1083;y=1578;z=-500;
    points.push_back(cv::Point3f(x,y,z));

    x=2000;y=155;z=1097;
    points.push_back(cv::Point3f(x,y,z));

    x=2000;y=155;z=-1097;
    points.push_back(cv::Point3f(x,y,z));

    for(unsigned int i = 0; i < points.size(); ++i)
    {
        std::cout << points[i] << std::endl;
    }

    return points;
}

double roll, pitch, yaw;

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    //tf2::Quaternion quat_tf;
    //geometry_msgs::Quaternion quat_msg = msg->pose.pose.orientation;

    //tf2::convert(quat_msg , quat_tf);
    //tf::Quaternion q = msg->pose.pose.orientation;

    //tf::Matrix3x3 m(quat_tf);

    //m.getRPY(roll, pitch, yaw);

    tf::Quaternion q(
            msg->pose.pose.orientation.x,
            msg->pose.pose.orientation.y,
            msg->pose.pose.orientation.z,
            msg->pose.pose.orientation.w);
    tf::Matrix3x3 m(q);
    //double roll, pitch, yaw;
    m.getRPY(roll, pitch, yaw);
    roll = roll*180/CV_PI;
    pitch = pitch*180/CV_PI;
    yaw = yaw*180/CV_PI;
    std::cerr << "FROM ODOM Roll: " << roll << ", Pitch: " << pitch << ", Yaw: " << yaw << std::endl;
}

class BasePublisher : public AbstractImageConverter
{

private:

    ros::Publisher publisher_markers, publisher_beacons, publisher_beacons_q, publisher_markers_q, publisher_ros_markers, publisher_odom;
    bool windowsEnabled;

protected:

    void process(const cv_bridge::CvImagePtr& cv_ptr)
    {
    	auv_common::Euler msg_odom;

    	msg_odom.roll = roll;
    	msg_odom.pitch = pitch;
    	msg_odom.yaw = yaw;

    	//cv::Point2f coordinates_sorted;


    	publisher_odom.publish(msg_odom);


        cv::Mat src = cv_ptr->image;
        //cv::resize(src, src, cv::Size(), 0.5, 0.5);
        cv::Mat src_copy1, src_copy2, src_copy3, src_copy4, src_copy5;
        src_copy1 = src.clone();
        src_copy2 = src.clone();
        src_copy3 = src.clone();
        src_copy4 = src.clone();
        src_copy5 = src.clone();

        //if (src.empty()) std::cerr<<"Empty";

        //ros::Subscriber sub = n.subscribe("chatter", 1000, chatterCallback);

        auv_common::Base msg_markers;
        auv_common::Base msg_beacons;
        geometry_msgs::Pose msg_pose_markers;
        geometry_msgs::Pose msg_pose_beacons;

        bool simulation_mode = false;
        cv::Mat camera_matrix, dist_coeffs;
        simulation_mode = true; /// For Simulator only !!!!

        //cv::resize(src, src, cv::Size(), 0.5, 0.5);

        std::vector<cv::Point2f> points2D, points2D_sorted, points2D_centred;
        std::vector<cv::Point2f> pointsTriangle;
        std::vector<cv::Vec4f> lines;
        /// Detection of beacons
        ///*
        BeaconDescriptor beacon_detection_image = beaconDetector.detect(src, true); /// Constructor
        if (beacon_detection_image.hasBeacons()) {

            std::cerr<<"Beacons detected!!!"<<std::endl;

            /*
            for (int i = 0; i < beacon_detection_image.getAllBeacons().size() - 1; i++) {
                for (int j = i + 1; j < beacon_detection_image.getAllBeacons().size(); j++) {
                    cv::line(src_copy1, beacon_detection_image.getCenter(i), beacon_detection_image.getCenter(j), (0, 255, 255), 1);
                }
            }
            */

            //cv::namedWindow("Detected Beacons");
            //if (!src_copy1.empty()) cv::imshow("Detected Beacons", src_copy1);

            for(size_t i = 0; i < beacon_detection_image.getAllBeacons().size(); i++) {
                int radius = cvRound(beacon_detection_image.getAllBeacons()[i][2]);
                cv::circle(src_copy2, beacon_detection_image.getCenter(i), 1, cv::Scalar(0,255,255), -1);
                cv::circle(src_copy2, beacon_detection_image.getCenter(i), radius, cv::Scalar(0,0,255), 1);
            }

            //cv::namedWindow("Circles Final");
            //if (!src_copy2.empty()) cv::imshow("Circles Final", src_copy2);

            //if (windowsEnabled) {
                //if (!src_copy2.empty()) cv::imshow(OPENCV_WINDOW_CIRCLES, src_copy2);
                //cv::waitKey(3);
            //}


            std::cout<<"Coordinates: "<<std::endl;

            for (int i = 0; i < beacon_detection_image.getAllBeacons().size(); i++) {
                std::cout<<i+1<<": "<<beacon_detection_image.getCenter(i)<<"; ";
                //cv::Point2f coordinates = convertToCentralCoordinates(beacon_detection_image.getCenter(i), src.size().width, src.size().height);
                points2D.push_back(beacon_detection_image.getCenter(i));
                points2D_sorted.push_back(beacon_detection_image.getCenter(i));
            }
            std::cout<<std::endl;


            std::cout<<"Coordinates in vector points2D: "<<std::endl;
            for (int i = 0; i < points2D.size(); ++i)
            {
                std::cout << points2D[i] << std::endl;
            }


            if (beacon_detection_image.allBeacons) {

                std::cout << "ALL MARKERS" << std::endl;
                std::array<float, LINES_NUMBER> parallel_lines_indxs;

                point_sorting(points2D_sorted, points2D_sorted.size(), true); /// Sort by y coordinate
                for (int i = 1; i < points2D_sorted.size(); ++i) {
                    if ((points2D_sorted[i-1].x > points2D_sorted[i].x) && (i != 4) && (i != 5) && (i != 2) && (i != 7)) {
                        cv::Point2f fake;
                        fake = points2D_sorted[i-1];
                        points2D_sorted[i-1] = points2D_sorted[i];
                        points2D_sorted[i] = fake;
                    }
                }

                /// Print Numbers
                for (int i = 0; i < points2D_sorted.size(); ++i) {
                    cv::putText(src_copy2, // Target image
                                std::to_string(i), /// Text
                                points2D_sorted[i], /// Position
                                cv::FONT_HERSHEY_DUPLEX,
                                1.0,
                                CV_RGB(118, 185, 0), //font color
                                2);
                }

                //cv::namedWindow("Beacons Numbers");
                //if (!src_copy2.empty()) cv::imshow("Beacons Numbers", src_copy2);

                if (windowsEnabled) {
                    if (!src_copy2.empty()) cv::imshow(OPENCV_WINDOW_BEACONS_NUMBERS, src_copy2);
                    cv::waitKey(3);
                }

                std::cout<<"Coordinates after sorting "<<std::endl;
                for (int i = 0; i < points2D_sorted.size(); ++i)
                {
                    std::cout << points2D_sorted[i] << std::endl;
                    points2D_centred.push_back(convertToCentralCoordinates(points2D_sorted[i], src.size().width, src.size().height));
                    //coordinates_sorted = convertToCentralCoordinates(points2D_sorted[i], src.size().width, src.size().height);
                }



                int counter = 0;

                for (int i = 0; i < beacon_detection_image.getAllBeacons().size() - 1; i++) {
                    for (int j = i + 1; j < beacon_detection_image.getAllBeacons().size(); j++) {
                        cv::Vec4f line = getLineFromPoints(beacon_detection_image.getCenter(i), beacon_detection_image.getCenter(j));
                        lines.push_back(line); /// To fill vector with lines
                    }
                }

                std::vector<std::vector<float>> lines_properties(3, std::vector<float>(lines.size())); /// Array for properties (slopes and lengths)

                for (int i = 0; i < lines.size(); i++) {
                    lines_properties [0][i] = i;
                    lines_properties [1][i] = getLineSlope(lines[i]);
                    lines_properties [2][i] = getLength(lines[i]);
                }

                sorting(lines_properties, 1, 0, lines.size() - 1); /// Sort slopes (from small to big ones)

                /// Print array
                ///*
                std::cout << "Array:" << std::endl;
                if (lines.size() > 0) {
                    for (int i = 0; i <= 2; i++) {
                        for (int j = 0; j < lines.size(); j++)
                        {
                            std::cout<<lines_properties [i][j]<<" ";
                        }
                        std::cout<<std::endl;
                    }
                }
                std::cout << std::endl;
                //*/

                /// Uncomment this if you want to use Above Station Navigation
                /*
                float max_length = -1;
                bool parallel = true;
                int index_parallel = -1;
                counter = 0;
                if ((lines.size() > 0) && (!station_detected)) {
                    for (int j = 1; j < lines.size(); j++) {
                        /// If lines parallel
                        if ((abs(lines_properties [1][j-1] - lines_properties [1][j]) < PARALLEL_CRITERIA) && (abs(lines_properties [2][j-1] - lines_properties [2][j]) < LENGTH_CRITERIA)) {
                            if (parallel) {
                                parallel = false;
                                parallel_lines_indxs[counter] = j-1;
                                counter++;
                            }
                        }
                        else {
                            parallel = true;
                        }
                    }
                }

                if (counter == 0) {
                    std::cerr << "No Parallel Lines" << std::endl;
                    /// Message
                }

                std::cout << "Indxs:" << std::endl;
                for (int j = 0; j < counter; j++) {
                    std::cout << parallel_lines_indxs[j] << " ";
                }
                std::cout << std::endl;

                if (counter > 0) {
                    for (int j = 0; j < counter; j++) {
                        if (lines_properties [2][parallel_lines_indxs[j]] > max_length) {
                            max_length = lines_properties [2][parallel_lines_indxs[j]];
                            index_parallel = parallel_lines_indxs[j];
                            std::cout << "Max Length: " << max_length << std::endl;
                        }
                    }
                    int counter_err_detector = 1;
                    for (int j = 0; j < lines.size(); j++) {
                        if ((abs(lines_properties [1][index_parallel] - lines_properties [1][j]) < PARALLEL_CRITERIA) && ((abs(lines_properties [2][index_parallel] - lines_properties [2][j]) < LENGTH_CRITERIA)) && (j != index_parallel)) {
                            counter_err_detector++;
                            if (counter_err_detector == 2) {
                                cv::Point2f p1, p2, p3, p4;
                                p1.x = lines[lines_properties [0][index_parallel]][0];
                                p1.y = lines[lines_properties [0][index_parallel]][1];
                                p2.x = lines[lines_properties [0][index_parallel]][2];
                                p2.y = lines[lines_properties [0][index_parallel]][3];
                                p3.x = lines[lines_properties [0][j]][0];
                                p3.y = lines[lines_properties [0][j]][1];
                                p4.x = lines[lines_properties [0][j]][2];
                                p4.y = lines[lines_properties [0][j]][3];
                                cv::line(src_copy4, p1, p2, cv::Scalar(0, 255, 0), 1);
                                cv::line(src_copy4, p3, p4, cv::Scalar(0, 255, 0), 1);

                                if (getDistance(p1.x, p1.y, p3.x, p3.y) < getDistance(p1.x, p1.y, p4.x, p4.y)) {
                                    cv::line(src_copy4, p1, p3, cv::Scalar(0, 255, 0), 1);
                                    cv::line(src_copy4, p2, p4, cv::Scalar(0, 255, 0), 1);
                                    bool flag1, flag2;
                                    flag1 = false;
                                    flag2 = false;
                                    cv::Point2f p5, p6, p7, p8;
                                    for (int i = 0; i < lines.size(); i++) {
                                        if ((abs(lines_properties [1][i] - getLineSlope(getLineFromPoints(p1, p3))) < PARALLEL_CRITERIA) && (index_parallel != i) && (j != i) && (abs(getLength(getLineFromPoints(p1, p3)) - lines_properties [2][i]) > 1)) {

                                            if (getLength(getLineFromPoints(p1, p3)) < lines_properties [2][i]) {
                                                p5.x = lines[lines_properties [0][i]][0];
                                                p5.y = lines[lines_properties [0][i]][1];
                                                p6.x = lines[lines_properties [0][i]][2];
                                                p6.y = lines[lines_properties [0][i]][3];
                                                cv::line(src_copy4, p5, p6, cv::Scalar(0, 0, 255), 1);
                                                flag1 = true;
                                            }
                                            else {
                                                p7.x = lines[lines_properties [0][i]][0];
                                                p7.y = lines[lines_properties [0][i]][1];
                                                p8.x = lines[lines_properties [0][i]][2];
                                                p8.y = lines[lines_properties [0][i]][3];
                                                cv::line(src_copy4, p7, p8, cv::Scalar(255, 0, 0), 1);
                                                flag2 = true;
                                            }
                                        }
                                    }
                                    if (flag1 && flag2) {
                                        for (int k = 0; k < points2D.size(); k++) {
                                            if ((points2D[k] != p1) && (points2D[k] != p2) && (points2D[k] != p3) && (points2D[k] != p4) && (points2D[k] != p5) && (points2D[k] != p6) && (points2D[k] != p7) && (points2D[k] != p8) && (points2D[k].x != 0) && (points2D[k].y != 0)) {
                                                cv::line(src_copy4, points2D[k], p7, cv::Scalar(0, 0, 255), 1);
                                                cv::line(src_copy4, points2D[k], p8, cv::Scalar(0, 0, 255), 1);
                                                std::cout << "POINT: " << points2D[k] << std::endl;
                                                std::cout<<"Coordinates p"<<p1 << p2 << p3 << p4 << p5 << p6 << p7 << p8<<std::endl;
                                            }
                                        }
                                    }
                                }
                                else {
                                    cv::line(src_copy4, p1, p4, cv::Scalar(0, 255, 0), 1);
                                    cv::line(src_copy4, p2, p3, cv::Scalar(0, 255, 0), 1);
                                    bool flag1, flag2;
                                    flag1 = false;
                                    flag2 = false;
                                    cv::Point2f p5, p6, p7, p8;
                                    for (int i = 0; i < lines.size(); i++) {
                                        if ((abs(lines_properties [1][i] - getLineSlope(getLineFromPoints(p1, p4))) < PARALLEL_CRITERIA) && (index_parallel != i) && (j != i) && (abs(getLength(getLineFromPoints(p1, p4)) - lines_properties [2][i]) > 1)) {

                                            if (getLength(getLineFromPoints(p1, p4)) < lines_properties [2][i]) {
                                                p5.x = lines[lines_properties [0][i]][0];
                                                p5.y = lines[lines_properties [0][i]][1];
                                                p6.x = lines[lines_properties [0][i]][2];
                                                p6.y = lines[lines_properties [0][i]][3];
                                                cv::line(src_copy4, p5, p6, cv::Scalar(0, 0, 255), 1);
                                                flag1 = true;
                                            }
                                            else {
                                                p7.x = lines[lines_properties [0][i]][0];
                                                p7.y = lines[lines_properties [0][i]][1];
                                                p8.x = lines[lines_properties [0][i]][2];
                                                p8.y = lines[lines_properties [0][i]][3];
                                                cv::line(src_copy4, p7, p8, cv::Scalar(255, 0, 0), 1);
                                                flag2 = true;
                                            }
                                        }
                                    }
                                    if (flag1 && flag2) {
                                        for (int k = 0; k < points2D.size(); k++) {
                                            if ((points2D[k] != p1) && (points2D[k] != p2) && (points2D[k] != p3) && (points2D[k] != p4) && (points2D[k] != p5) && (points2D[k] != p6) && (points2D[k] != p7) && (points2D[k] != p8) && (points2D[k].x != 0) && (points2D[k].y != 0)) {
                                                cv::line(src_copy4, points2D[k], p7, cv::Scalar(0, 0, 255), 1);
                                                cv::line(src_copy4, points2D[k], p8, cv::Scalar(0, 0, 255), 1);
                                                std::cout << "POINT: " << points2D[k] << std::endl;
                                                std::cout<<"Coordinates p"<<p1 << p2 << p3 << p4 << p5 << p6 << p7 << p8<<std::endl;
                                            }
                                        }
                                    }
                                }
                                //break;
                            }
                        }
                    }
                }

                cv::namedWindow("Detected Lines");
                if (!src_copy4.empty()) cv::imshow("Detected Lines", src_copy4);

                */

                //sorting(lines_properties, 2, 0, lines.size() - 1); /// Sort lengths (from small to big ones)

                /// Compute pose
                if (simulation_mode) {
                    double focal_length_sim = src.cols; // Approximate focal length.
                    cv::Point2d center_image_sim = cv::Point2d(src.cols/2,src.rows/2);
                    camera_matrix = (cv::Mat_<double>(3,3) << focal_length_sim, 0, center_image_sim.x, 0 , focal_length_sim, center_image_sim.y, 0, 0, 1);
                    dist_coeffs = cv::Mat::zeros(4,1,cv::DataType<double>::type); // Assuming no lens distortion
                }
                else {
                    cv::FileStorage fs("../out_camera_data.xml", cv::FileStorage::READ);

                    fs["camera_matrix"] >> camera_matrix;
                    fs["distortion_coefficients"] >> dist_coeffs;
                }

                std::cout << "camera_matrix\n" << camera_matrix << std::endl;
                std::cout << "\ndist coeffs\n" << dist_coeffs << std::endl;

                cv::Mat rvecsb(3,1,cv::DataType<double>::type);
                cv::Mat tvecsb(3,1,cv::DataType<double>::type);

                std::vector<cv::Point3f> objectPoints = Generate3DPoints();

                cv::solvePnP(objectPoints, points2D_sorted, camera_matrix, dist_coeffs, rvecsb, tvecsb);

                std::cerr << "Pose using Beacons" << std::endl;

                cv::Mat rotCamerMatrix;

                cv::Rodrigues(rvecsb,rotCamerMatrix);
                cv::Vec3d eulerAngles;
                getEulerAngles(rotCamerMatrix, eulerAngles);

                float yaw   = eulerAngles[2]-90;
                float pitch = (-1)*eulerAngles[1];
                float roll  = eulerAngles[0]+180;

                std::cerr << "YAW: " << yaw << std::endl;
                std::cerr << "PITCH: " << pitch << std::endl;
                std::cerr << "ROLL: " << roll << std::endl;

                std::cerr << "Rotation: " << rvecsb << std::endl;
                std::cerr << "Translation: " << tvecsb << std::endl;

                std::vector<float> array_for_mat = tvecsb.reshape(0, 1);

                msg_beacons.x = array_for_mat[0];
                msg_beacons.y = array_for_mat[1];
                msg_beacons.z = array_for_mat[2];
                msg_beacons.roll = roll;
                msg_beacons.pitch = pitch;
                msg_beacons.yaw = yaw;

                publisher_beacons.publish(msg_beacons);

                cv::Mat rot;
                cv::Rodrigues(rvecsb, rot);

                tf::Matrix3x3 tf_rot(rot.at<double>(0,0), rot.at<double>(0,1), rot.at<double>(0,2),
                                     rot.at<double>(1,0), rot.at<double>(1,1), rot.at<double>(1,2),
                                     rot.at<double>(2,0), rot.at<double>(2,1), rot.at<double>(2,2));

                tf::Vector3 tf_transl(tvecsb.at<double>(0,0), tvecsb.at<double>(1,0), tvecsb.at<double>(2,0));

                tf::Quaternion quat;
                tf_rot.getRotation(quat);

                /*
                tf::Quaternion q(quat.x, quat.y, quat.z, quat.w);
                tf::Matrix3x3 m(q);
                double roll, pitch, yaw;
                m.getRPY(roll, pitch, yaw);
                std::cout << "Roll: " << roll << ", Pitch: " << pitch << ", Yaw: " << yaw << std::endl;
                */

                msg_pose_beacons.position.x = tf_transl[0];
                msg_pose_beacons.position.y = tf_transl[1];
                msg_pose_beacons.position.z = tf_transl[2];

                msg_pose_beacons.orientation.x = quat[0];
                msg_pose_beacons.orientation.y = quat[1];
                msg_pose_beacons.orientation.z = quat[2];
                msg_pose_beacons.orientation.w = quat[3];

                publisher_beacons_q.publish(msg_pose_beacons);
            }
        }
        else {
            std::cerr<<"NO Beacons detected!!!"<<std::endl;
        }
        //*/

        std::vector<cv::Vec3d> rvecsm, tvecsm;
        bool hasMarker;
        //if (!src_copy3.empty()) cv::cvtColor(src_copy3, src_copy3, CV_BGR2GRAY);

        ArUcoDetector.pose_Estimation(hasMarker, src_copy3, rvecsm, tvecsm);

        /// To find camera pose
        cv::Mat R, rotCamerMatrix;
        cv::Mat tvec, rvec;
        cv::Vec3d tvecv, rvecv;

        std::cerr << "Pose using ArUco" << std::endl;
        float yaw_average (0), pitch_average (0), roll_average (0);

        for (int i = 0; i < rvecsm.size(); i++) {
            std::cerr << i << ": " << std::endl;
            tvec = DoubleMatFromVec3b(tvecsm[i]);
            rvec = DoubleMatFromVec3b(rvecsm[i]);

            cv::Rodrigues(rvecsm[i], R); /// Calculate object pose R matrix (Converts a rotation matrix to a rotation vector or vice versa)
            R = R.t();                   /// Calculate camera R matrix
            cv::Rodrigues(R, rvecv);     /// Calculate camera Rotation vector
            tvec = -R*tvec;              /// Calculate camera Translation vector

            //rvec = DoubleMatFromVec3b(rvecv);

            cv::Rodrigues(rvecv,rotCamerMatrix);
            cv::Vec3d eulerAngles;
            getEulerAngles(rotCamerMatrix, eulerAngles);

            float yaw   = eulerAngles[1];
            float pitch = eulerAngles[0];
            float roll  = eulerAngles[2];

            std::cerr << "YAW: " << yaw << std::endl;
            std::cerr << "PITCH: " << pitch << std::endl;
            std::cerr << "ROLL: " << roll << std::endl;

            std::cerr << "Rotation: " << rvecv << std::endl;
            std::cerr << "Translation: " << tvec << std::endl;

            yaw_average = yaw_average + yaw;
            pitch_average = pitch_average + pitch;
            roll_average = roll_average + roll;

            std::vector<float> array_for_mat = tvec.reshape(0, 1);

            msg_markers.x = array_for_mat[0];
            msg_markers.y = array_for_mat[1];
            msg_markers.z = array_for_mat[2];


            //std::cerr << "Rotation: " << rvecsm[i] << std::endl;
            //std::cerr << "Translation: " << tvecsm[i] << std::endl;
        }

        if (rvecsm.size() != 0) {
            yaw_average = yaw_average/rvecsm.size();
            pitch_average = pitch_average/rvecsm.size();
            roll_average = roll_average/rvecsm.size();

            msg_markers.roll = roll_average;
            msg_markers.pitch = pitch_average;
            msg_markers.yaw = yaw_average;
        }

        publisher_markers.publish(msg_markers);
    }

public:

    /// Constructor
    BasePublisher(const std::string& inputImageTopic, bool enableWindows) : AbstractImageConverter(inputImageTopic),
                                                                                   windowsEnabled(enableWindows) {

        image_transport::ImageTransport it(nodeHandle);
        //linesImagePublisher = it.advertise("/frontCamMat/image_lines", 100);
        //maskedImagePublisher = it.advertise("/frontCamMat/image_masked", 100);

        publisher_beacons = nodeHandle.advertise<auv_common::Base>(BASE_PUBLISH_TOPIC_BEACONS, 100);
        publisher_markers = nodeHandle.advertise<auv_common::Base>(BASE_PUBLISH_TOPIC_MARKERS, 100);
        publisher_beacons_q = nodeHandle.advertise<geometry_msgs::Pose>(BASE_PUBLISH_TOPIC_BEACONS_Q, 100);
        //publisher_markers_q = nodeHandle.advertise<auv_common::Base>(BASE_PUBLISH_TOPIC_MARKERS_Q, 100);
        //publisher_ros_markers = nodeHandle.advertise<auv_common::Base>(BASE_PUBLISH_TOPIC_MARKERS_ROS, 100);
        publisher_odom = nodeHandle.advertise<auv_common::Euler>(BASE_PUBLISH_TOPIC_ODOM, 100);

        if (windowsEnabled) {
            cv::namedWindow(OPENCV_WINDOW, CV_WINDOW_AUTOSIZE);
            cv::namedWindow(OPENCV_WINDOW_BEACONS_NUMBERS, CV_WINDOW_AUTOSIZE);
        }

    }

    /// Destructor
    ~BasePublisher()
    {
        if (windowsEnabled) {
            cv::destroyWindow(OPENCV_WINDOW);
            cv::destroyWindow(OPENCV_WINDOW_BEACONS_NUMBERS);
        }
    }

};


int main(int argc, char **argv)
{

#if CV_MAJOR_VERSION == 2
    ROS_INFO("%s", "OpenCV version = 2");
#elif CV_MAJOR_VERSION == 3
    ROS_INFO("%s", "OpenCV version = 3");
#endif

    ros::init(argc, argv, BASE_LOCATOR_NODE_NAME);
    ros::NodeHandle nodeHandle(BASE_LOCATOR_NODE_NAME);

    bool windowsEnabled;
    nodeHandle.param(ENABLE_WINDOWS_PARAM, windowsEnabled, false);

    BasePublisher publisher_beacons(CAMERA_BOTTOM_TOPIC, windowsEnabled);
    BasePublisher publisher_markers(CAMERA_FRONT_TOPIC, windowsEnabled);
    BasePublisher publisher_beacons_q(CAMERA_BOTTOM_TOPIC, windowsEnabled);
    BasePublisher publisher_markers_q(CAMERA_FRONT_TOPIC, windowsEnabled);
    BasePublisher publisher_ros_markers(CAMERA_BOTTOM_TOPIC, windowsEnabled);
    BasePublisher publisher_odom(CAMERA_BOTTOM_TOPIC, windowsEnabled);

    ros::Subscriber sub_odom = nodeHandle.subscribe("/odom", 100, odomCallback);

    //ros::Publisher odom_pub = nodeHandle.advertise<auv_common::Euler>("base/odom/angle", 100);

    ros::spin();

    return 0;
}


