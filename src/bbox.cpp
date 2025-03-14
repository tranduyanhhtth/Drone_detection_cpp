#include "bbox.h"

cv::Mat Bbox::draw_box(const string &frame_link,
                       const vector<Detection> &detection)
{
    cv::Mat image_draw = cv::imread(frame_link);

    if (!detection.empty())
    {
        for (const auto &bbox : detection)
        {
            float x_center = bbox.x;
            float y_center = bbox.y;
            float w = bbox.w;
            float h = bbox.h;
            string label = bbox.label;
            float conf = bbox.confidence;

            // Draw bounding box
            float x = x_center - w / 2;
            float y = y_center - h / 2;
            float x_max = x_center + w / 2;
            float y_max = y_center + h / 2;

            cv::circle(image_draw, cv::Point(static_cast<int>(x_center), static_cast<int>(y_center)), 2, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(image_draw, cv::Point(static_cast<int>(x), static_cast<int>(y)), cv::Point(static_cast<int>(x_max), static_cast<int>(y_max)), cv::Scalar(0, 255, 0), 1);
            cv::putText(image_draw, cv::format("%s: %.2f", label.c_str(), conf), cv::Point(static_cast<int>(x), static_cast<int>(y) - 2), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
            cv::putText(image_draw, cv::format("Center_box: (%d, %d)", static_cast<int>(x_center), static_cast<int>(y_center)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
    }
    else
    {
        cv::putText(image_draw, cv::format("Center_image: (%d, %d)", image_draw.cols / 2, image_draw.rows / 2), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }

    cv::circle(image_draw, cv::Point(image_draw.cols / 2, image_draw.rows / 2), 2, cv::Scalar(0, 255, 255), 2);
    cv::putText(image_draw, cv::format("Center_image: (%d, %d)", image_draw.cols / 2, image_draw.rows / 2), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

    return image_draw;
}

cv::Mat Bbox::draw_box(cv::Mat image, const vector<Detection> &detections)
{
    if (image.empty())
    {
        std::cerr << "Input image is empty" << std::endl;
        return cv::Mat();
    }

    cv::Mat image_draw = image.clone(); // Create a copy to draw on

    if (!detections.empty())
    {
        for (const auto &bbox : detections)
        {
            float x_center = bbox.x;
            float y_center = bbox.y;
            float w = bbox.w;
            float h = bbox.h;
            string label = bbox.label;
            float conf = bbox.confidence;

            // Draw bounding box
            float x = x_center - w / 2;
            float y = y_center - h / 2;
            float x_max = x_center + w / 2;
            float y_max = y_center + h / 2;

            cv::circle(image_draw, cv::Point(static_cast<int>(x_center), static_cast<int>(y_center)), 2, cv::Scalar(0, 255, 0), 2);
            cv::rectangle(image_draw, cv::Point(static_cast<int>(x), static_cast<int>(y)), cv::Point(static_cast<int>(x_max), static_cast<int>(y_max)), cv::Scalar(0, 255, 0), 1);
            cv::putText(image_draw, cv::format("%s: %.2f", label.c_str(), conf), cv::Point(static_cast<int>(x), static_cast<int>(y) - 2), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 255), 2);
            cv::putText(image_draw, cv::format("Center_box: (%d, %d)", static_cast<int>(x_center), static_cast<int>(y_center)), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        }
    }
    else
    {
        cv::putText(image_draw, cv::format("Center_image: (%d, %d)", image_draw.cols / 2, image_draw.rows / 2), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }

    cv::circle(image_draw, cv::Point(image_draw.cols / 2, image_draw.rows / 2), 2, cv::Scalar(0, 255, 255), 2);
    cv::putText(image_draw, cv::format("Center_image: (%d, %d)", image_draw.cols / 2, image_draw.rows / 2), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

    return image_draw;
}