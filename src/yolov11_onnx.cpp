#include "yolov11_onnx.h"
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <fstream>

using namespace std;

Yolov11_Onnx::Yolov11_Onnx(const string &onnx_model_path, const vector<string> &label_list,
                           const cv::Size &input_shape, float confidence_threshold, float nms_threshold)
    : onnx_model_path_(onnx_model_path),
      input_shape_(input_shape),
      confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold),
      label_list_(label_list),
      env_(ORT_LOGGING_LEVEL_WARNING, "Yolov11_Onnx"),
      session_(nullptr)
{
    Ort::SessionOptions session_options;

    session_options.SetIntraOpNumThreads(1); // So luong xu ly song song trong tung phep toan
    session_options.SetInterOpNumThreads(1); // So luong xu ly giua cac phep toan
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    /*
     * Load model từ file ONNX
     */
    session_ = Ort::Session(env_, onnx_model_path.c_str(), session_options);
}

Ort::Value Yolov11_Onnx::preprocessing(const cv::Mat &frame)
{
    int original_height = frame.rows;
    int original_width = frame.cols;
    this->resize_ratio_w_ = static_cast<double>(original_width) / input_shape_.width;
    this->resize_ratio_h_ = static_cast<double>(original_height) / input_shape_.height;

    cv::Mat input_img;
    cv::resize(frame, input_img, cv::Size(640, 640), 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(input_img, input_img, cv::COLOR_BGR2RGB);
    input_img.convertTo(input_img, CV_32F, 1.0 / 255.0);

    /* Prepare input data in CHW format */
    const size_t channel_size = static_cast<size_t>(input_shape_.height) * input_shape_.width;
    input_data.resize(3 * channel_size);

    for (int c = 0; c < 3; ++c)
    {
        for (int h = 0; h < input_shape_.height; ++h)
        {
            for (int w = 0; w < input_shape_.width; ++w)
            {
                input_data[c * channel_size + h * input_shape_.width + w] = input_img.at<cv::Vec3f>(h, w)[c];
            }
        }
    }

    /* Define tensor shape */
    const array<int64_t, 4> input_shape_arr{1, 3, input_shape_.height, input_shape_.width};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape_arr.data(),
        input_shape_arr.size());

    /**********************************/
    // ofstream out_file("/home/danz/Downloads/Drone_detect/yolov11_cpp_project/input_tensor_data.txt");
    // if (!out_file.is_open())
    // {
    //     cerr << "Không thể mở file để ghi!" << endl;
    //     return tensor;
    // }

    // out_file << "Tensor Shape: 1 3 " << input_shape_.height << " " << input_shape_.width << "\n";
    // out_file << "Total Elements: " << input_data.size() << "\n\n";
    // out_file << "Input Tensor Values:\n";

    // for (size_t i = 0; i < input_data.size(); ++i)
    // {
    //     out_file << input_data[i];
    //     if (i < input_data.size() - 1)
    //     {
    //         out_file << " "; // Thêm dấu cách giữa các giá trị
    //     }
    //     if ((i + 1) % 10 == 0)
    //     {
    //         out_file << "\n"; // Xuống dòng sau mỗi 10 giá trị để dễ đọc
    //     }
    // }
    // out_file << "\n";

    // out_file.close();
    // cout << "Đã ghi dữ liệu input_tensor vào file input_tensor_data.txt" << endl;
    /*********************************/

    cout << "First value: " << input_data[0] << ", Size: " << input_data.size() << endl;
    return tensor;
}

vector<Detection> Yolov11_Onnx::postprocessing(const vector<Ort::Value> &output_tensor)
{
    const float *output_data = output_tensor[0].GetTensorData<float>();
    auto type_info = output_tensor[0].GetTensorTypeAndShapeInfo();
    vector<int64_t> output_shape = type_info.GetShape();

    int num_attributes = output_shape[1];
    int num_boxes = output_shape[2];

    vector<float> x_center(num_boxes);
    vector<float> y_center(num_boxes);
    vector<float> w(num_boxes);
    vector<float> h(num_boxes);
    vector<vector<float>> confidence(num_attributes - 4, vector<float>(num_boxes));

    for (int i = 0; i < num_boxes; ++i)
    {
        x_center[i] = output_data[0 * num_attributes * num_boxes + 0 * num_boxes + i];
        y_center[i] = output_data[0 * num_attributes * num_boxes + 1 * num_boxes + i];
        w[i] = output_data[0 * num_attributes * num_boxes + 2 * num_boxes + i];
        h[i] = output_data[0 * num_attributes * num_boxes + 3 * num_boxes + i];
        for (int j = 4; j < num_attributes; ++j)
        {
            confidence[j - 4][i] = output_data[0 * num_attributes * num_boxes + j * num_boxes + i];
        }
    }

    /* class_id & max_class_prob */
    vector<int> class_id(num_boxes);
    vector<float> max_class_prob(num_boxes);
    for (int i = 0; i < num_boxes; ++i)
    {
        float max_prob = -1.0f;
        int max_idx = -1;
        for (int j = 0; j < num_attributes - 4; ++j)
        {
            if (confidence[j][i] > max_prob)
            {
                max_prob = confidence[j][i];
                max_idx = j;
            }
        }
        class_id[i] = max_idx;
        max_class_prob[i] = max_prob;
    }

    /* Filter base on confidence threshold */
    vector<Detection> detections;
    for (int i = 0; i < num_boxes; ++i)
    {
        if (max_class_prob[i] > this->confidence_threshold_)
        {
            Detection det;
            det.x = x_center[i] * this->resize_ratio_w_;
            det.y = y_center[i] * this->resize_ratio_h_;
            det.w = w[i] * this->resize_ratio_w_;
            det.h = h[i] * this->resize_ratio_h_;
            det.confidence = max_class_prob[i];
            det.label = "Drone";
            detections.push_back(det);
        }
    }

    /* NMS */
    if (!detections.empty())
    {
        vector<cv::Rect> boxes;
        vector<float> scores;
        for (const auto &det : detections)
        {
            float x1 = det.x - det.w / 2;
            float y1 = det.y - det.h / 2;
            float x2 = det.x + det.w / 2;
            float y2 = det.y + det.h / 2;
            boxes.push_back(cv::Rect(
                static_cast<int>(x1),
                static_cast<int>(y1),
                static_cast<int>(x2 - x1),
                static_cast<int>(y2 - y1)));
            scores.push_back(det.confidence);
        }

        vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, this->confidence_threshold_, this->nms_threshold_, indices);

        vector<Detection> final_detections;
        for (int idx : indices)
        {
            final_detections.push_back(detections[idx]);
        }
        detections = final_detections;
    }

    cout << "Detected objects:\n";
    for (const auto &det : detections)
    {
        cout << "[" << det.x << ", " << det.y << ", "
             << det.w << ", " << det.h << ", \"" << det.label
             << "\", " << det.confidence << "]\n";
    }
    return detections;
}

vector<Detection> Yolov11_Onnx::detect(const string &image_path)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_.GetInputNameAllocated(0, allocator);
    vector<const char *> input_names{input_name.get()};

    vector<string> output_names;
    vector<const char *> output_names_cstr;
    size_t output_count = session_.GetOutputCount();

    for (size_t i = 0; i < output_count; ++i)
    {
        auto output_name = session_.GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        output_names_cstr.push_back(output_names.back().c_str());
    }

    cv::Mat frame = cv::imread(image_path);
    Ort::Value input_tensor = preprocessing(frame);

    // {
    //     float *input_data = input_tensor.GetTensorMutableData<float>();
    //     auto tensor_info = input_tensor.GetTensorTypeAndShapeInfo();
    //     auto tensor_shape = tensor_info.GetShape();
    //     size_t total_elements = tensor_info.GetElementCount();

    //     ofstream out_file("/home/danz/Downloads/Drone_detect/yolov11_cpp_project/input_tensor_data_1.txt");
    //     if (!out_file.is_open())
    //     {
    //         cerr << "Không thể mở file để ghi!" << endl;
    //     }
    //     else
    //     {
    //         out_file << "Tensor Shape: ";
    //         for (size_t i = 0; i < tensor_shape.size(); ++i)
    //         {
    //             out_file << tensor_shape[i];
    //             if (i < tensor_shape.size() - 1)
    //                 out_file << " x ";
    //         }
    //         out_file << "\nTotal Elements: " << total_elements << "\n\n";

    //         out_file << "Input Tensor Values:\n";
    //         for (size_t i = 0; i < total_elements; ++i)
    //         {
    //             out_file << input_data[i];
    //             if (i < total_elements - 1)
    //                 out_file << " ";
    //             if ((i + 1) % 10 == 0)
    //                 out_file << "\n";
    //         }
    //         out_file << "\n";
    //         out_file.close();
    //         cout << "Đã ghi dữ liệu input_tensor vào file input_tensor_data.txt" << endl;
    //     }
    // }

    vector<Ort::Value> output_tensors = session_.Run(Ort::RunOptions{nullptr},
                                                     input_names.data(),
                                                     &input_tensor,
                                                     1,
                                                     output_names_cstr.data(),
                                                     output_names_cstr.size());
    /*********************************************************************/
    const float *output_data = output_tensors[0].GetTensorData<float>();
    cout << "First 100 output values: ";
    for (int i = 0; i < 100; ++i)
    {
        cout << output_data[i] << " ";
    }
    cout << endl;
    /*********************************************************************/
    vector<Detection> result = postprocessing(output_tensors);
    return result;
}

vector<Detection> Yolov11_Onnx::detect(const cv::Mat &image)
{
    // Kiểm tra xem image có hợp lệ không
    if (image.empty())
    {
        cerr << "Input image is empty!" << endl;
        return vector<Detection>();
    }

    // Chuẩn bị tên đầu vào và đầu ra cho ONNX Runtime
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session_.GetInputNameAllocated(0, allocator);
    vector<const char *> input_names{input_name.get()};

    vector<string> output_names;
    vector<const char *> output_names_cstr;
    size_t output_count = session_.GetOutputCount();
    for (size_t i = 0; i < output_count; ++i)
    {
        auto output_name = session_.GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());
        output_names_cstr.push_back(output_names.back().c_str());
    }

    // Tiền xử lý frame
    Ort::Value input_tensor = preprocessing(image);

    // Chạy inference
    vector<Ort::Value> output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names_cstr.data(),
        output_names_cstr.size());

    // Xử lý hậu kỳ và trả về kết quả
    vector<Detection> result = postprocessing(output_tensors);
    return result;
}