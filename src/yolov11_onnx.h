#ifndef YOLOV11_ONNX_H
#define YOLOV11_ONNX_H

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace std;

struct Detection
{
    float x;
    float y;
    float w;
    float h;
    string label;
    float confidence;
};

class Yolov11_Onnx
{
private:
    string onnx_model_path_;
    cv::Size input_shape_;
    float confidence_threshold_;
    float nms_threshold_;
    vector<string> label_list_;
    double resize_ratio_w_;
    double resize_ratio_h_;
    Ort::Env env_;
    Ort::Session session_;
    vector<float> input_data;

public:
    Yolov11_Onnx(const string &onnx_model_path,
                 const vector<string> &label_list = {"drone"},
                 const cv::Size &input_shape = cv::Size(640, 640),
                 float confidence_threshold = 0.7f,
                 float nms_threshold = 0.85f);

    ~Yolov11_Onnx() = default;

    Ort::Value preprocessing(const cv::Mat &frame);
    vector<Detection> postprocessing(const vector<Ort::Value> &output_tensor);
    vector<Detection> detect(const string &image_path);
    vector<Detection> detect(const cv::Mat &image);
};

#endif