#include "YOLOv11SegInference.h"
#include <opencv2/imgproc.hpp>
#include <numeric>
#include <iostream>

YOLOv11SegInference::YOLOv11SegInference(const std::string& model_path, float mean_[3], float std_[3])
    : mnn_executor(model_path, mean_, std_)
{
    mnn_executor.loadModel();
}

// ------------------------- Letterbox -------------------------
void YOLOv11SegInference::letterbox(const cv::Mat& src, cv::Mat& dst, cv::Size size,
                                    std::pair<int,int>& pad_info, float& scale)
{
    int w0 = src.cols, h0 = src.rows;
    int iw = size.width, ih = size.height;
    scale = std::min(float(iw)/w0, float(ih)/h0);
    int nw = int(w0 * scale);
    int nh = int(h0 * scale);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh));

    dst = cv::Mat(ih, iw, CV_8UC3, cv::Scalar(114,114,114));
    int pad_x = (iw - nw) / 2;
    int pad_y = (ih - nh) / 2;
    resized.copyTo(dst(cv::Rect(pad_x, pad_y, nw, nh)));

    pad_info = {pad_x, pad_y};
}

cv::Mat YOLOv11SegInference::preprocess(const cv::Mat& img,
                                        std::pair<int,int>& pad_info,
                                        float& scale)
{
    // --- Step 1: Letterbox resize with padding 114 ---
    int h0 = img.rows;
    int w0 = img.cols;
    int iw = input_size_, ih = input_size_;
    scale = std::min(float(iw)/w0, float(ih)/h0);
    int nw = int(w0 * scale);
    int nh = int(h0 * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));

    int pad_x = (iw - nw) / 2;
    int pad_y = (ih - nh) / 2;

    cv::Mat img_letterbox(ih, iw, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(img_letterbox(cv::Rect(pad_x, pad_y, nw, nh)));

    pad_info = {pad_x, pad_y};

    // --- Step 2: BGR -> RGB ---
    cv::cvtColor(img_letterbox, img_letterbox, cv::COLOR_BGR2RGB);

    // --- Step 3: Convert to float and normalize ---
    cv::Mat img_float;
    img_letterbox.convertTo(img_float, CV_32F, 1.0/255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(img_float, channels);

    // --- Step 4: HWC -> CHW ---
    cv::Mat chw(3, ih*iw, CV_32F);
    for(int c=0; c<3; c++){
        std::memcpy(chw.ptr<float>(c), channels[c].data, ih*iw*sizeof(float));
    }

    // --- Step 5: reshape to (1,3,ih,iw) ---
    cv::Mat input_tensor = chw.reshape(1, {1,3,ih,iw});

    return input_tensor;
}



// ------------------------- NMS -------------------------
void YOLOv11SegInference::NMSBoxes(const std::vector<cv::Rect2f>& boxes,
                                   const std::vector<float>& scores,
                                   float score_threshold,
                                   float iou_threshold,
                                   std::vector<int>& indices)
{
    std::vector<int> idxs(boxes.size());
    std::iota(idxs.begin(), idxs.end(), 0);

    idxs.erase(std::remove_if(idxs.begin(), idxs.end(),
        [&](int i){ return scores[i] < score_threshold; }), idxs.end());

    std::sort(idxs.begin(), idxs.end(),
        [&](int i1,int i2){ return scores[i1] > scores[i2]; });

    while(!idxs.empty()){
        int i = idxs[0];
        indices.push_back(i);
        std::vector<int> tmp;
        for(size_t k=1;k<idxs.size();++k){
            int j = idxs[k];
            float inter_x1 = std::max(boxes[i].x, boxes[j].x);
            float inter_y1 = std::max(boxes[i].y, boxes[j].y);
            float inter_x2 = std::min(boxes[i].x + boxes[i].width, boxes[j].x + boxes[j].width);
            float inter_y2 = std::min(boxes[i].y + boxes[i].height, boxes[j].y + boxes[j].height);
            float w = std::max(0.0f, inter_x2 - inter_x1);
            float h = std::max(0.0f, inter_y2 - inter_y1);
            float inter = w*h;
            float ovr = inter/(boxes[i].area() + boxes[j].area() - inter);
            if(ovr <= iou_threshold) tmp.push_back(j);
        }
        idxs = tmp;
    }
}

// ------------------------- Mask 生成 -------------------------
cv::Mat YOLOv11SegInference::processMask(const std::vector<float>& proto_flat,
                                         const std::vector<float>& mask_coef,
                                         const std::pair<int,int>& pad_info,
                                         float scale,
                                         const cv::Size& orig_size)
{
    const int proto_c = 32, ph = 160, pw = 160;
    cv::Mat proto_map(ph, pw, CV_32FC(proto_c));

    for(int y=0;y<ph;y++){
        for(int x=0;x<pw;x++){
            for(int c=0;c<proto_c;c++){
                proto_map.at<cv::Vec<float,32>>(y,x)[c] = proto_flat[c*ph*pw + y*pw + x];
            }
        }
    }

    cv::Mat mask_low = cv::Mat::zeros(ph, pw, CV_32F);
    int K = mask_coef.size();
    for(int y=0;y<ph;y++){
        for(int x=0;x<pw;x++){
            float val = 0.f;
            cv::Vec<float,32> v = proto_map.at<cv::Vec<float,32>>(y,x);
            for(int k=0;k<std::min(K, proto_c);k++) val += v[k]*mask_coef[k];
            mask_low.at<float>(y,x) = 1.0f/(1.0f + std::exp(-val));
        }
    }

    int x0 = pad_info.first;
    int y0 = pad_info.second;
    int x1 = x0 + int(orig_size.width*scale + 0.5f);
    int y1 = y0 + int(orig_size.height*scale + 0.5f);

    x0 = int(float(x0)*pw/input_size_);
    x1 = int(float(x1)*pw/input_size_);
    y0 = int(float(y0)*ph/input_size_);
    y1 = int(float(y1)*ph/input_size_);

    x0 = std::clamp(x0,0,pw-1);
    y0 = std::clamp(y0,0,ph-1);
    x1 = std::clamp(x1,x0+1,pw);
    y1 = std::clamp(y1,y0+1,ph);

    cv::Rect roi(x0,y0,x1-x0,y1-y0);
    cv::Mat mask_cropped = mask_low(roi);

    cv::Mat mask_resized;
    cv::resize(mask_cropped, mask_resized, orig_size, 0,0, cv::INTER_LINEAR);

    cv::Mat mask_bin;
    cv::threshold(mask_resized, mask_bin, 0.5, 255, cv::THRESH_BINARY);
    mask_bin.convertTo(mask_bin, CV_8U);

    return mask_bin;
}

// ------------------------- Postprocess -------------------------
SegOut YOLOv11SegInference::postprocess(const std::vector<std::vector<std::vector<float>>>& outputs,
                                        const cv::Mat& orig_img,
                                        const std::pair<int,int>& pad_info,
                                        float scale)
{
    SegOut out;
    if(outputs.empty()) return out;

    const auto& det_out = outputs[0][0];    // [1,116,8400] flatten
    const auto& proto_out = outputs[0][1];  // [1,32,160,160] flatten

    int num_features = 116;
    int num_preds = det_out.size()/num_features;

    std::vector<std::vector<float>> dets(num_preds, std::vector<float>(num_features,0));
    for(int j=0;j<num_preds;j++)
        for(int i=0;i<num_features;i++)
            dets[j][i] = det_out[i*num_preds + j];

    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> mask_coefs;

    for(const auto& det : dets){
        float cx = det[0], cy = det[1], w = det[2], h = det[3];
        float max_score = -1000.f; int cls_id=0;
        for(int i=0;i<num_classes_;i++){
            if(det[4+i] > max_score)
            {
                max_score=det[4+i];
                cls_id=i; 
            }
        }
        if(max_score < conf_thres_) continue;

        float x0 = pad_info.first;
        float y0 = pad_info.second;

        // cx/cy/w/h 是模型输出，假设是 640x640 输入尺度
        float scale_inv = 1.0f / scale;

        float x1 = (cx - w/2 - x0) * scale_inv;
        float y1 = (cy - h/2 - y0) * scale_inv;
        float x2 = (cx + w/2 - x0) * scale_inv;
        float y2 = (cy + h/2 - y0) * scale_inv;

        x1 = std::clamp(x1, 0.f, float(orig_img.cols));
        y1 = std::clamp(y1, 0.f, float(orig_img.rows));
        x2 = std::clamp(x2, 0.f, float(orig_img.cols));
        y2 = std::clamp(y2, 0.f, float(orig_img.rows));

        boxes.push_back(cv::Rect2f(x1,y1,x2-x1,y2-y1));

        scores.push_back(max_score);
        class_ids.push_back(cls_id);

        std::vector<float> coef(det.begin()+4+num_classes_, det.end());
        mask_coefs.push_back(coef);
    }

    std::vector<int> keep;
    NMSBoxes(boxes,scores,conf_thres_,iou_thres_,keep);

    cv::Size orig_size = orig_img.size();
    for(int idx: keep){
        Detection det;
        det.box = boxes[idx];
        det.score = scores[idx];
        det.class_id = class_ids[idx];
        det.mask = processMask(proto_out, mask_coefs[idx], pad_info, scale, orig_size);
        out.detections.push_back(det);
    }

    return out;
}

// ------------------------- getYOLOsegResult -------------------------
bool YOLOv11SegInference::getYOLOsegResult(const cv::Mat& img, SegOut& out)
{
    std::pair<int,int> pad_info;
    float scale;
    cv::Mat input_tensor = preprocess(img, pad_info, scale);

    std::vector<cv::Mat> inputs = {input_tensor};
    std::vector<std::vector<std::vector<float>>> outputs;
    std::vector<std::pair<std::string,std::vector<int>>> out_shapes;
    if(!mnn_executor.runInference(inputs, outputs, out_shapes)) return false;

    out = postprocess(outputs, img, pad_info, scale);
    return true;
}
