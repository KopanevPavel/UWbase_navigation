#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>

/**
#if CUDA_ENABLED == 0
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif
*/

#include "../../include/beacons/BeaconDetector.h"
#include "../../include/beacons/BeaconDescriptor.h"

const int NUMBER_OF_BEACONS = 9;

/// Uncomment this if you want to detect colors using trackbars
const cv::String window_detection_name = "After Color Filtering";

const int max_value_H = 360/2;
const int max_value = 255;
int low_H = 0, low_S = 0, low_V = 0;
int default_low_H = 11, default_high_H = 96, default_low_S = 95;
int high_H = max_value_H, high_S = max_value, high_V = max_value;

static void on_low_H_thresh_trackbar(int, void *) {
    low_H = cv::min(high_H-1, low_H);
    cv::setTrackbarPos("Low H", window_detection_name, low_H);
}
static void on_high_H_thresh_trackbar(int, void *) {
    high_H = cv::max(high_H, low_H+1);
    cv::setTrackbarPos("High H", window_detection_name, high_H);
}
static void on_low_S_thresh_trackbar(int, void *) {
    low_S = cv::min(high_S-1, low_S);
    cv::setTrackbarPos("Low S", window_detection_name, low_S);
}
static void on_high_S_thresh_trackbar(int, void *) {
    high_S = cv::max(high_S, low_S+1);
    cv::setTrackbarPos("High S", window_detection_name, high_S);
}
static void on_low_V_thresh_trackbar(int, void *) {
    low_V = cv::min(high_V-1, low_V);
    cv::setTrackbarPos("Low V", window_detection_name, low_V);
}
static void on_high_V_thresh_trackbar(int, void *) {
    high_V = cv::max(high_V, low_V+1);
    cv::setTrackbarPos("High V", window_detection_name, high_V);
}

void createTrackbars_filter() {
    /// Create trackbars and insert them into window to change H,S,V values
    cv::namedWindow(window_detection_name);

    /// Trackbars to set thresholds for HSV values
    cv::createTrackbar("Low H", window_detection_name, &low_H, max_value_H, on_low_H_thresh_trackbar);
    cv::createTrackbar("High H", window_detection_name, &high_H, max_value_H, on_high_H_thresh_trackbar);
    cv::createTrackbar("Low S", window_detection_name, &low_S, max_value, on_low_S_thresh_trackbar);
    cv::createTrackbar("High S", window_detection_name, &high_S, max_value, on_high_S_thresh_trackbar);
    cv::createTrackbar("Low V", window_detection_name, &low_V, max_value, on_low_V_thresh_trackbar);
    cv::createTrackbar("High V", window_detection_name, &high_V, max_value, on_high_V_thresh_trackbar);

}

/// Uncomment this if you want to apply morphology using trackbars
static void on_Trackbar(int, void *) {};
int dilation_size = 0;
int erosion_size = 0;
int erosion_elem = 0;
int dilation_elem = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

const cv::String window_erosion_name = "After Erosion";
const cv::String window_dilation_name = "After Dilation";

void createTrackbars_morphology() {
    /// Create windows for trackbars
    cv::namedWindow(window_erosion_name);
    cv::namedWindow(window_dilation_name);

    /// Create Erosion Trackbar
    cv::createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", window_erosion_name,
                       &erosion_elem, max_elem,
                       on_Trackbar);

    cv::createTrackbar("Kernel size:\n n +1", window_erosion_name,
                       &erosion_size, max_kernel_size,
                       on_Trackbar);

    /// Create Dilation Trackbar
    cv::createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", window_dilation_name,
                       &dilation_elem, max_elem,
                       on_Trackbar);

    cv::createTrackbar("Kernel size:\n 2n +1", window_dilation_name,
                       &dilation_size, max_kernel_size,
                       on_Trackbar);

}

/// Uncomment this if you want to use mouse color picker (it finds color range of an object which you clicked at)
/*
cv::Mat image_for_color_pick_1;
cv::Mat image_for_color_pick_2;

/// Mouse callback function (returns color of the place where you clicked)
void pick_color(int event, int x, int y, int f, void *)
{
    if (event==1)
    {
        int r = 3;
        int off[9*2] = {0,0, -r,-r, -r,0, -r,r, 0,r, r,r, r,0, r,-r, 0,-r};
        for (int i=0; i < 9; i++)
        {
            cv::Vec3b p = image_for_color_pick_1.at<cv::Vec3b>(y+off[2*i], x+off[2*i+1]);
            std::cerr << int(p[0]) << " " << int(p[1]) << " " << int(p[2]) << std::endl;
            image_for_color_pick_2.push_back(p);
        }
    }
}
*/

void CLAHE_correction(const cv::Mat &src, cv::Mat &dst) {

    cv::Mat lab_image;
    cv::cvtColor(src, lab_image, CV_BGR2Lab);

    /// Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);  // Now we have the L image in lab_planes[0]

    /// Apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(lab_planes[0], dst);

    /// Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_image);

    /// Convert back to RGB
    cv::Mat image_clahe;
    cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);

    dst = image_clahe;
}

void BeaconDetector::extractGreenColour(const cv::Mat &src, cv::Mat &dst) {
    cv::blur(src, dst, cv::Size(3,3));
    //cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0);

    /// Uncomment this if you want to use CLAHE correction
    //CLAHE_correction(dst, dst);

    cv::Mat hsv;

    /// Convert from BGR to HSV colorspace
    if (!dst.empty()) cv::cvtColor(dst, hsv, CV_BGR2HSV);

    //cv::namedWindow("Filtering src");
    //if (!hsv.empty()) cv::imshow("Filtering src", hsv);

    /// This (using color picker function)
    /*
    image_for_color_pick_1 = hsv;

    cv::namedWindow("Pick Green");
    if (!src.empty()) cv::imshow("Pick Green", src);

    cv::setMouseCallback("Pick Green", pick_color, 0);

    cv::Scalar m, v;
    cv::meanStdDev(image_for_color_pick_2, m, v);
    std::cerr << "mean, var (GREEN): " << std::endl;
    std::cerr << m[0] << " " << m[1] << " " << m[2] << " " << v[0] << " " << v[1] << " " << v[2] << std::endl;

    cv::Scalar lower_green(m[0]-v[0], m[1]-v[1], m[2]-v[2]); /// Mean - var for low
    cv::Scalar higher_green(m[0]+v[0], m[1]+v[1], m[2]+v[2]); /// Mean + var for high
    */

    //cv::Scalar lower_green(78, 95, 0); /// Mean - var for low
    //cv::Scalar higher_green(180, 255, 146); /// Mean + var for high

    /// https://www.tydac.ch/color/

    /// Or This (using trackbars)
    createTrackbars_filter();
    std::array<int, 3> lower_green = {low_H, low_S, low_V}; //H_S_V
    std::array<int, 3> higher_green = {high_H, high_S, high_V};

    /// Detect the object based on HSV Range Values
    cv::Mat mask;
    cv::inRange(hsv, lower_green, higher_green, mask);
    cv::bitwise_and(hsv, hsv, dst, mask = mask); /// dst - image with applied mask

    dst = mask;
    if (!dst.empty()) {cv::imshow(window_detection_name, dst); cv::waitKey(3);}
    //if (!dst.empty()) {cv::namedWindow("After Color Filtering"); cv::imshow("After Color Filtering", dst);}
}

void BeaconDetector::morphology(const cv::Mat &src, cv::Mat &dst) {
    cv::Mat element;

    /// Uncomment this if you want to apply morphology (erosion and dilation) using trackbars
    createTrackbars_morphology();

    int erosion_type;

    if (erosion_elem == 0) {erosion_type = cv::MORPH_RECT;}
    else if (erosion_elem == 1) {erosion_type = cv::MORPH_CROSS;}
    else if (erosion_elem == 2) {erosion_type = cv::MORPH_ELLIPSE;}

    element = cv::getStructuringElement(erosion_type,
                                        cv::Size(erosion_size + 1, erosion_size+1),
                                        cv::Point(erosion_size, erosion_size));

    /// Apply the erosion operation
    cv::erode(src, dst, element);
    //if (!dst.empty()) cv::imshow(window_erosion_name, dst);

    int dilation_type;

    if (dilation_elem == 0) {dilation_type = cv::MORPH_RECT;}
    else if (dilation_elem == 1) {dilation_type = cv::MORPH_CROSS;}
    else if (dilation_elem == 2) {dilation_type = cv::MORPH_ELLIPSE;}

    element = cv::getStructuringElement(dilation_type,
                                        cv::Size(2*dilation_size + 1, 2*dilation_size+1),
                                        cv::Point(dilation_size, dilation_size));

    /// Apply the dilation operation
    cv::dilate(dst, dst, element);
    //if (!dst.empty()) cv::imshow(window_dilation_name, dst);

    /// Apply morphologyEx
    int size = 2;
    element = cv::getStructuringElement(CV_SHAPE_RECT, cv::Size(2*size+1, 2*size+1));
    cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, element);
    if (!dst.empty()) {cv::namedWindow("After morphologyEx"); cv::imshow("After morphologyEx", dst); cv::waitKey(3);}
}

std::vector<cv::Vec3f> BeaconDetector::findCircles(const cv::Mat& src) {
    std::vector<cv::Vec3f> circles;

    cv::Mat gray, src_copy, dst;

    cv::blur(src, dst, cv::Size(3,3));
    //cv::GaussianBlur(src, dst, cv::Size(0, 0), 2);
    //cv::namedWindow("After GaussianBlur");
    //if (!dst.empty()) cv::imshow("After GaussianBlur", dst);

    //if (!dst.empty()) cv::cvtColor(dst, gray, CV_BGR2GRAY);
    gray = dst.clone();

    /// Apply the Hough Transform to find the circles
    cv::HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, 10, 100, 15, 0, 60);

    src_copy = src.clone();

    /// Draw the circles detected
    for(size_t i = 0; i < circles.size(); i++) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::circle(src_copy, center, 3, cv::Scalar(0,255,255), -1);
        cv::circle(src_copy, center, radius, cv::Scalar(0,0,255), 1);
    }

    //cv::namedWindow("Circles");
    //if (!src_copy.empty()) cv::imshow("Circles", src_copy);

    return circles;
}

BeaconDescriptor BeaconDetector::detect(const cv::Mat src, bool withPreprocess) {
    cv::Mat image;
    if (withPreprocess) {
        if (!src.empty()) extractGreenColour(src, image);
        else {
            std::cerr<<"No Beacons Were Detected"<<std::endl;
            return BeaconDescriptor::noBeacons();
        }
        if (!image.empty()) morphology(image, image);
        else {
            std::cerr<<"No Beacons Were Detected"<<std::endl;
            return BeaconDescriptor::noBeacons();
        }
    } else
        image = src.clone();

    if (!image.empty()) {
        std::vector<cv::Vec3f> detectedCircles = findCircles(image);
        if (detectedCircles.size() == NUMBER_OF_BEACONS) {
            std::cerr<<"All Beacons Were Detected"<<std::endl;
            return BeaconDescriptor::create(true, detectedCircles);
        }
        else if (detectedCircles.size() == 0) {
            std::cerr<<"No Beacons Were Detected"<<std::endl;
            return BeaconDescriptor::noBeacons();
        }
        else {
            std::cerr<<"Not All Beacons Were Detected"<<std::endl;
            return BeaconDescriptor::create(false, detectedCircles);
        }
    }
    else {
        std::cerr<<"No Beacons Were Detected"<<std::endl;
        return BeaconDescriptor::noBeacons();
    }
}

