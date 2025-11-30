// MNNInfer.h
#ifndef MNN_INFER_H
#define MNN_INFER_H

#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>
#include <MNN/Tensor.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <memory>

class MNNInfer 
{
    public:
        MNNInfer(std::string modelPath,float mean_[3],float std_[3]); 
        ~MNNInfer();

    public:
        int loadModel();
        bool runInference(std::vector<cv::Mat> &inputs, 
                         std::vector<std::vector<std::vector<float>>> &outputs,
                         std::vector<std::pair<std::string, std::vector<int>>> &output_shapes);

    private:
        std::string m_modelPath;
        std::shared_ptr<MNN::Interpreter> m_net;
        MNN::Session* m_session = nullptr;
        MNN::Tensor* m_inputTensor = nullptr;
        float mnn_mean[3];
        float mnn_std[3];

};

#endif // MNN_INFER_H