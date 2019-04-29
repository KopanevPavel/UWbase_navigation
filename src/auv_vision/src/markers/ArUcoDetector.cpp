#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "../../include/markers/ArUcoDetector.h"

void ArUcoDetector::pose_Estimation(bool& hasMarker, const cv::Mat& src, std::vector<cv::Vec3d>& rvecs, std::vector<cv::Vec3d>& tvecs) {

    int wait_time = 10;
    float actual_marker_length = 0.101;  // this should be in meters

    cv::Mat image, image_copy;
    cv::Mat camera_matrix, dist_coeffs;
    std::ostringstream vector_to_marker;
    bool simulation_mode = false;
    simulation_mode = true; /// For Simulator only !!!!

    cv::Ptr<cv::aruco::Dictionary> dictionary =
            cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

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

    image = src.clone();
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f> > corners;
    cv::aruco::detectMarkers(image, dictionary, corners, ids);

    if (!image.empty()) {cv::namedWindow("Detected markers"); cv::imshow("Detected markers", image); cv::waitKey(3);}

    // if at least one marker detected
    if ((ids.size() > 0) && (!image.empty()) && (!corners.empty()))
    {
        cv::aruco::drawDetectedMarkers(image, corners, ids);
        cv::aruco::estimatePoseSingleMarkers(corners, actual_marker_length,
                                             camera_matrix, dist_coeffs, rvecs, tvecs);
        // draw axis for each marker
        for(int i=0; i < ids.size(); i++)
        {
            cv::aruco::drawAxis(image, camera_matrix, dist_coeffs,
                                rvecs[i], tvecs[i], 0.1);

            /// To draw text - coordinates on image
            /*
            vector_to_marker.str(std::string());
            vector_to_marker << std::setprecision(4)
                             << "x: " << std::setw(8) <<  tvecs[0](0);
            cv::putText(image, vector_to_marker.str(),
                        cvPoint(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cvScalar(0, 252, 124), 1, CV_AA);

            vector_to_marker.str(std::string());
            vector_to_marker << std::setprecision(4)
                             << "y: " << std::setw(8) << tvecs[0](1);
            cv::putText(image, vector_to_marker.str(),
                        cvPoint(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cvScalar(0, 252, 124), 1, CV_AA);

            vector_to_marker.str(std::string());
            vector_to_marker << std::setprecision(4)
                             << "z: " << std::setw(8) << tvecs[0](2);
            cv::putText(image, vector_to_marker.str(),
                        cvPoint(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cvScalar(0, 252, 124), 1, CV_AA);
            */
        }


        if (!image.empty()) {cv::namedWindow("Pose estimation"); cv::imshow("Pose estimation", image); cv::waitKey(3);}
        hasMarker = true;
    } else {
        hasMarker = false;
    }

}