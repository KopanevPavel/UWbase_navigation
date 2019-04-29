#ifndef AUV_VISION_ARUCODETECTOR_H
#define AUV_VISION_ARUCODETECTOR_H

#include <opencv2/opencv.hpp>

class ArUcoDetector {

private:

public:

    ArUcoDetector() = default;
    ~ArUcoDetector() = default;
    ArUcoDetector& operator=(const ArUcoDetector& other) = default;

    void pose_Estimation(bool& hasMarker, const cv::Mat& src, std::vector<cv::Vec3d>& rvecs, std::vector<cv::Vec3d>& tvecs);

};

#endif //AUV_VISION_ARUCODETECTOR_H
