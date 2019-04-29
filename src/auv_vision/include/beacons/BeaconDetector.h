#ifndef AUV_VISION_BEACONDETECTOR_H
#define AUV_VISION_BEACONDETECTOR_H

#include <opencv2/opencv.hpp>
#include "BeaconDescriptor.h"

class BeaconDetector {

private:

    void extractGreenColour(const cv::Mat& src, cv::Mat& dst);
    std::vector<cv::Vec3f> findCircles(const cv::Mat& src);
    void morphology(const cv::Mat& src, cv::Mat& dst);

public:

    BeaconDetector() = default;
    ~BeaconDetector() = default;
    BeaconDetector& operator=(const BeaconDetector& other) = default;

    BeaconDescriptor detect(const cv::Mat src, bool withPreprocess);

};

#endif //AUV_VISION_BEACONDETECTOR_H



