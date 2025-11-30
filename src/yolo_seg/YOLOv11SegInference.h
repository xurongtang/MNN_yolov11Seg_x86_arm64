#ifndef YOLOV11SEG_INFERENCE_H
#define YOLOV11SEG_INFERENCE_H

#include <InferMNN/mnnInfer.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

struct Detection {
    cv::Rect_<float> box;      // x1, y1, x2, y2 in original image
    float score;
    int class_id;
    cv::Mat mask;              // binary mask same size as original image
};

struct SegOut {
    std::vector<Detection> detections;
};

class YOLOv11SegInference {
public:
    YOLOv11SegInference(const std::string& model_path, float mean_[3], float std_[3]);
    ~YOLOv11SegInference() = default;

    // 推理
    bool getYOLOsegResult(const cv::Mat& img, SegOut& out);

private:
    MNNInfer mnn_executor;
    const int num_classes_ = 80;
    const float conf_thres_ = 0.25f;   // OBJ_THRESH
    const float iou_thres_  = 0.45f;   // NMS_THRESH
    const int input_size_ = 640;

    // ------------------------- 前处理 -------------------------
    cv::Mat preprocess(const cv::Mat& img,
                       std::pair<int,int>& pad_info,
                       float& scale);

    void letterbox(const cv::Mat& src, cv::Mat& dst, cv::Size size,
                   std::pair<int,int>& pad_info, float& scale);

    // ------------------------- 后处理 -------------------------
    SegOut postprocess(const std::vector<std::vector<std::vector<float>>>& outputs,
                       const cv::Mat& orig_img,
                       const std::pair<int,int>& pad_info,
                       float scale);

    void NMSBoxes(const std::vector<cv::Rect2f>& boxes,
                  const std::vector<float>& scores,
                  float score_threshold,
                  float iou_threshold,
                  std::vector<int>& indices);

    cv::Mat processMask(const std::vector<float>& proto_flat,
                        const std::vector<float>& mask_coef,
                        const std::pair<int,int>& pad_info,
                        float scale,
                        const cv::Size& orig_size);

    float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};

#endif // YOLOV11SEG_INFERENCE_H
