#ifndef AUV_VISION_BEACONDESCRIPTOR_H
#define AUV_VISION_BEACONDESCRIPTOR_H

#include <opencv2/opencv.hpp>

class BeaconDescriptor {

private:

    bool severalBeacons;
    std::vector<cv::Vec3f> beacons;

    BeaconDescriptor(bool allBeacons, bool severalBeacons, const std::vector<cv::Vec3f>& beacons); /// Constructor

public:

    static BeaconDescriptor noBeacons();
    static BeaconDescriptor create(bool allBeacons, const std::vector<cv::Vec3f>& beacons);  /// Functions to initialise class

    BeaconDescriptor(const BeaconDescriptor& other);
    ~BeaconDescriptor() = default;
    BeaconDescriptor& operator=(const BeaconDescriptor& other);

    std::vector<cv::Vec3f> getAllBeacons();
    bool hasBeacons();

    cv::Point2f getCenter(const int i);

    bool allBeacons;
};

#endif //AUV_VISION_BEACONDESCRIPTOR_H
