#pragma once

#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
#include "autoaim_interfaces/msg/detection_array.hpp"

class GreenLightDetector {
public:
    GreenLightDetector(
        float minArea = 200.0f,
        float minCircularity = 0.7f,
        int minBrightness = 240 * 3
    );

    void set_input_image(const cv::Mat image);
    void detect();
    std::vector<autoaim_interfaces::msg::Detection> get_detection_arr() const;
    cv::Mat debug_draw_armors();

private:
    cv::Mat preprocess() const;
    bool validateContour(const std::vector<cv::Point>& contour) const;

    cv::Mat img;
    float min_area_;
    float min_circularity_;
    int min_brightness_;

    std::vector<autoaim_interfaces::msg::Detection> detection_arr;
};

// Implementation
GreenLightDetector::GreenLightDetector(float minArea, float minCircularity, int minBrightness)
    : min_area_(minArea)
    , min_circularity_(minCircularity)
    , min_brightness_(minBrightness)
{}

void GreenLightDetector::set_input_image(const cv::Mat image)  {
    this->img = image;
}

cv::Mat GreenLightDetector::preprocess() const {
    cv::Mat hsv, mask;
    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(40, 120, 60), cv::Scalar(80, 255, 200), mask);
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, 
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15,15)));
    return mask;
}

bool GreenLightDetector::validateContour(const std::vector<cv::Point>& contour) const {
    float area = cv::contourArea(contour);
    if (area < min_area_) return false;

    float perimeter = cv::arcLength(contour, true);
    float circularity = (4 * CV_PI * area) / (perimeter * perimeter);
    return circularity >= min_circularity_;
}

void GreenLightDetector::detect() {
    cv::Mat green_mask = preprocess();
    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(green_mask, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    detection_arr.clear();    

    for (int i = 0; i < contours.size(); i++) {
        if (hierarchy[i][0] == -1 && hierarchy[i][1] == -1 && 
            hierarchy[i][2] == -1 && hierarchy[i][3] != -1) {
            
            if (!validateContour(contours[i])) continue;

            cv::RotatedRect ellipse = cv::fitEllipse(contours[i]);
            cv::Point2f center = ellipse.center;
            
            cv::Vec3b pixel = img.at<cv::Vec3b>(center.y, center.x);
            if (pixel[0] + pixel[1] + pixel[2] < min_brightness_) continue;

            cv::Point2f u(center.x, center.y - ellipse.size.height / 2);
            cv::Point2f d(center.x, center.y + ellipse.size.height / 2);
            cv::Point2f l(center.x - ellipse.size.width / 2, center.y);
            cv::Point2f r(center.x + ellipse.size.width / 2, center.y);
            
            autoaim_interfaces::msg::Detection detection;
            detection.color = 0;
            detection.label = 0;
            detection.confidence = 1.0f;
            detection.tl.x = l.x; detection.tl.y = l.y;
            detection.bl.x = d.x; detection.bl.y = d.y;
            detection.br.x = r.x; detection.br.y = r.y;
            detection.tr.x = u.x; detection.tr.y = u.y;
            detection_arr.push_back(detection);
            return ;
        }
    }
    return ;
}

std::vector<autoaim_interfaces::msg::Detection> GreenLightDetector::get_detection_arr() const {
    return detection_arr;
}

cv::Mat GreenLightDetector::debug_draw_armors() {
    for (const auto& detection : detection_arr) {
        cv::Point2f kpts[4] {
            cv::Point2f(detection.tl.x, detection.tl.y),
            cv::Point2f(detection.bl.x, detection.bl.y),
            cv::Point2f(detection.br.x, detection.br.y),
            cv::Point2f(detection.tr.x, detection.tr.y)
        };
        for (int j = 0; j < 4; j++) {
            line(img, kpts[j], kpts[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
        }
        putText(img, "Green Light", kpts[0] - cv::Point2f(0, 15), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    }
    return img;
}