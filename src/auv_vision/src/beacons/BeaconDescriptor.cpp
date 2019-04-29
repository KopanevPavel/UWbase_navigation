#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "../../include/beacons/BeaconDescriptor.h"

BeaconDescriptor BeaconDescriptor::noBeacons(){return BeaconDescriptor(false, false, std::vector<cv::Vec3f>());}

BeaconDescriptor BeaconDescriptor::create(bool allBeacons, const std::vector<cv::Vec3f>& beacons) {
    if (allBeacons) return BeaconDescriptor(true, false, beacons);
    else return BeaconDescriptor(false, true, beacons);
}

BeaconDescriptor::BeaconDescriptor(bool allBeacons, bool severalBeacons, const std::vector<cv::Vec3f>& beacons) {
    this->allBeacons = allBeacons;
    this->severalBeacons = severalBeacons;
    this->beacons = beacons;
}

BeaconDescriptor::BeaconDescriptor(const BeaconDescriptor &other) {
    this->allBeacons = other.allBeacons;
    this->severalBeacons = other.severalBeacons;
    this->beacons = other.beacons;
}

BeaconDescriptor& BeaconDescriptor::operator=(const BeaconDescriptor &other) {
    if (this != &other) {
        this->allBeacons = other.allBeacons;
        this->severalBeacons = other.severalBeacons;
        this->beacons = other.beacons;
    }
    return *this;
}

cv::Point2f BeaconDescriptor::getCenter(const int i) {return cv::Point2f(beacons[i][0], beacons[i][1]);}

std::vector<cv::Vec3f> BeaconDescriptor::getAllBeacons() {
    return beacons;
}

bool BeaconDescriptor::hasBeacons(){return (allBeacons || severalBeacons);}


