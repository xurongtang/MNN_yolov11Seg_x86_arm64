#include "yolo_seg/YOLOv11SegInference.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <filesystem>

// 为 OpenCV 3/4 兼容定义
#ifndef CV_VERSION_EPOCH
#define HAVE_COLOR_MASK
#endif

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] 
                  << " <model.mnn> <input_image.jpg> [output_dir]\n";
        std::cerr << "Example: " << argv[0] 
                  << " yolo11l-seg_x86.mnn bus.jpg ./results\n";
        return -1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];
    std::string outputDir = (argc > 3) ? argv[3] : "./yolo_seg_results";

    // 创建输出目录
    std::filesystem::create_directories(outputDir);

    // YOLO 归一化参数 (img / 255.0)
    float mean[3] = {0.0f, 0.0f, 0.0f};
    float std[3]  = {255.0f, 255.0f, 255.0f};

    try {
        // 1. 创建分割推理器
        YOLOv11SegInference seg_model(modelPath, mean, std);

        // 2. 读取图像
        cv::Mat img = cv::imread(imagePath);
        if (img.empty()) {
            std::cerr << "❌ Failed to load image: " << imagePath << std::endl;
            return -1;
        }
        std::cout << "✅ Loaded image: " << img.cols << "x" << img.rows << std::endl;

        // 3. 推理
        std::vector<cv::Mat> inputs = {img};
        std::vector<std::vector<std::vector<float>>> outputs;
        std::vector<std::pair<std::string, std::vector<int>>> out_shapes;

        // 4. 后处理
        SegOut result;
        bool res = seg_model.getYOLOsegResult(img, result);
        std::cout << "✅ Detected " << result.detections.size() << " instances." << std::endl;

        // 5. 可视化与保存
        cv::Mat img_draw = img.clone();
        cv::RNG rng(2025); // 固定随机种子保证颜色一致

        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& det = result.detections[i];

            // 生成随机颜色
            cv::Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

            // 绘制 bbox
            cv::rectangle(img_draw, det.box, color, 2);

            // 绘制标签
            std::string label = "cls" + std::to_string(det.class_id) + " " + 
                               std::to_string(static_cast<int>(det.score * 100)) + "%";
            int baseline = 0;
            cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point top_left = det.box.tl();
            cv::rectangle(img_draw,
                          cv::Point(top_left.x, top_left.y - label_size.height - 5),
                          cv::Point(top_left.x + label_size.width, top_left.y),
                          color, -1);
            cv::putText(img_draw, label, cv::Point(top_left.x, top_left.y - 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            // 保存二值掩码 (白色前景，黑色背景)
            cv::Mat mask_uint8;
            det.mask.convertTo(mask_uint8, CV_8U, 255.0);
            std::string mask_path = outputDir + "/mask_" + std::to_string(i) + ".png";
            // cv::imwrite(mask_path, mask_uint8);
            // std::cout << "  → Saved mask: " << mask_path << std::endl;

            // 可选：保存带掩码叠加的图像
#ifdef HAVE_COLOR_MASK
            // 创建彩色掩码
            cv::Mat colored_mask = cv::Mat::zeros(img.size(), CV_8UC3);
            colored_mask.setTo(color, mask_uint8);
            cv::addWeighted(img_draw, 1.0, colored_mask, 0.5, 0.0, img_draw);
#endif
        }

        // 保存带 bbox 和 mask 的结果图
        std::string result_path = outputDir + "/result.jpg";
        cv::imwrite(result_path, img_draw);
        std::cout << "✅ Saved result image: " << result_path << std::endl;

        // 6. 打印检测详情
        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& det = result.detections[i];
            if (det.mask.empty()) {
                std::cerr << "⚠️ Skip empty mask for detection " << i << std::endl;
                continue;
            }
            // std::cout << "Detection " << i << ": "
            //           << "class=" << det.class_id
            //           << ", score=" << det.score
            //           << ", box=[" << det.box.x << "," << det.box.y << "," 
            //           << det.box.width << "x" << det.box.height << "]"
            //           << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
