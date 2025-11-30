// MNNInfer.cpp (修复版)
#include "mnnInfer.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>

MNNInfer::MNNInfer(std::string modelPath,float mean_[3],float std_[3])
    : m_modelPath(modelPath) {
        for(int i = 0; i < 3; i++)
        {
            mnn_mean[i] = mean_[i];
            mnn_std[i] = std_[i];
        }
    }

MNNInfer::~MNNInfer() {
    if (m_session) {
        m_net->releaseSession(m_session);
    }
}

int MNNInfer::loadModel() {
    m_net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(m_modelPath.c_str()));
    if (!m_net) {
        std::cerr << "❌ Failed to load MNN model: " << m_modelPath << std::endl;
        return -1;
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    MNN::BackendConfig backendConfig;
    backendConfig.precision = MNN::BackendConfig::Precision_High;
    config.backendConfig = &backendConfig;

    m_session = m_net->createSession(config);
    if (!m_session) {
        std::cerr << "❌ Failed to create MNN session." << std::endl;
        return -1;
    }

    // 获取输入张量
    auto inputTensors = m_net->getSessionInputAll(m_session);
    if (inputTensors.empty()) {
        std::cerr << "❌ No input tensor found!" << std::endl;
        return -1;
    }
    std::string inputName = inputTensors.begin()->first;
    m_inputTensor = inputTensors.begin()->second;
    
    if (!m_inputTensor) {
        std::cerr << "❌ Failed to get input tensor." << std::endl;
        return -1;
    }

    // 打印输入信息
    auto shape = m_inputTensor->shape();
    std::cout << "✅ Model loaded. Input shape (NCHW): ";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

bool MNNInfer::runInference(std::vector<cv::Mat> &inputs,
                           std::vector<std::vector<std::vector<float>>> &outputs,
                           std::vector<std::pair<std::string, std::vector<int>>> &out_shapes) {
    if (!m_session || !m_inputTensor) return false;
    if (inputs.empty()) return false;

    outputs.clear();
    out_shapes.clear();

    for (size_t i = 0; i < inputs.size(); ++i) {
        cv::Mat& input_tensor = inputs[i];
        if (input_tensor.empty()) {
            outputs.push_back({});
            continue;
        }

        // --- Step 1: 直接复制已经预处理好的数据到 MNN 输入 tensor ---
        int total_size = input_tensor.total();
        std::memcpy(m_inputTensor->host<float>(), input_tensor.ptr<float>(), total_size * sizeof(float));

        // --- Step 2: 运行推理 ---
        auto start_time = std::chrono::high_resolution_clock::now();
        m_net->runSession(m_session);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "✅ Inference time for image " << i << ": " << duration.count() << " ms" << std::endl;

        // --- Step 3: 获取输出 ---
        auto outputNames = m_net->getSessionOutputAll(m_session);
        std::vector<std::vector<float>> image_outputs;

        for (const auto& item : outputNames) {
            auto outputTensor = m_net->getSessionOutput(m_session, item.first.c_str());
            MNN::Tensor outputUser(outputTensor, MNN::Tensor::CAFFE);
            outputTensor->copyToHostTensor(&outputUser);

            std::vector<int> tensor_shape;
            size_t total = 1;
            for (auto s : outputTensor->shape()) {
                total *= s;
                tensor_shape.push_back(static_cast<int>(s));
            }
            image_outputs.emplace_back(outputUser.host<float>(), outputUser.host<float>() + total);

            if (i == 0) {
                out_shapes.push_back({item.first, tensor_shape});
            }
        }

        outputs.push_back(std::move(image_outputs));
    }

    return true;
}
