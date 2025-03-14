#ifndef BBOX_H
#define BBOX_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "yolov11_onnx.h"

using namespace std;

class Bbox
{
public:
    static cv::Mat draw_box(const string &image_path,
                            const vector<Detection> &detections);
    static cv::Mat draw_box(cv::Mat image, const vector<Detection> &detections);
};

#endif