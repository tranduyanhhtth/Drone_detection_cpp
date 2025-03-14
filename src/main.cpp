#include <iostream>
#include <opencv2/opencv.hpp>
#include "yolov11_onnx.h"
#include "bbox.h"

int main()
{
    const string image_path = "/home/danz/Downloads/Drone_detect/Data_bla/DroneTestDataset/Drone_TestSet/VS_P7637.jpg";
    const string model_path = "/home/danz/Downloads/Drone_detect/src/best.onnx";

    Yolov11_Onnx detector(model_path);

    vector<Detection> detections = detector.detect(image_path);

    cv::Mat result_frame = Bbox::draw_box(image_path, detections);

    cv::imshow("Detection result", result_frame);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}