import MNN
import numpy as np
import cv2
from torchvision.ops import nms
import torch

FIXED_COLORS = [
    (255, 0, 0),      # 蓝
    (0, 255, 0),      # 绿
    (0, 0, 255),      # 红
    (255, 255, 0),    # 青
    (255, 0, 255),    # 紫
    (0, 255, 255),    # 黄
    (128, 0, 0),      # 深红
    (0, 128, 0),      # 深绿
    (0, 0, 128),      # 深蓝
    (128, 128, 0),    # 橄榄
    (128, 0, 128),    # 深紫
    (0, 128, 128),    # 深青
    (64, 64, 64),     # 深灰
    (192, 192, 192),  # 浅灰
    (255, 128, 0),    # 橙
    (128, 255, 0),    # 黄绿
    (0, 255, 128),    # 绿青
    (0, 128, 255),    # 蓝青
]

# ------------------------- 预处理 -------------------------
def preprocess_image(image_path, input_size=(640, 640)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h0, w0 = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ih, iw = input_size
    scale = min(iw / w0, ih / h0)
    nw, nh = int(w0 * scale), int(h0 * scale)

    resized = cv2.resize(img, (nw, nh))
    pad_x = (iw - nw) // 2
    pad_y = (ih - nh) // 2

    new_img = np.full((ih, iw, 3), 114, dtype=np.uint8)
    new_img[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    new_img = new_img.astype(np.float32) / 255.0
    new_img = np.transpose(new_img, (2, 0, 1))
    new_img = new_img[np.newaxis, :, :, :]

    return new_img, (h0, w0), scale, pad_x, pad_y

# ------------------------- Sigmoid -------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.72, 88.72)))

# ------------------------- YOLO 解码 -------------------------
def decode_yolo_seg(outputs, orig_shape, scale, pad_x, pad_y, input_size=640, conf_thres=0.5, iou_thres=0.45):
    det_out = outputs[0]  # (1, 116, 8400)
    proto_out = outputs[1]  # (1, 32, 160, 160)

    det_out = np.squeeze(det_out, axis=0)  # (116, 8400)
    proto_out = np.squeeze(proto_out, axis=0)  # (32, 160, 160)

    num_classes = 80
    det_out = det_out.T  # (8400, 116)

    boxes_xywh = det_out[:, :4]  # (8400, 4)
    score_cls = det_out[:, 4 : 4 + num_classes]  # (8400, 80)
    mask_coeffs = det_out[:, 4 + num_classes:]  # (8400, 31)

    max_scores = np.max(score_cls, axis=1)
    class_ids = np.argmax(score_cls, axis=1)

    mask = max_scores > conf_thres
    boxes_xywh = boxes_xywh[mask]
    scores_sel = max_scores[mask]
    class_ids = class_ids[mask]
    mask_coeffs = mask_coeffs[mask]

    if len(boxes_xywh) == 0:
        return [], [], [], proto_out, []

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_norm = np.stack([x1, y1, x2, y2], axis=1)

    boxes_t = torch.from_numpy(boxes_norm * input_size)
    scores_t = torch.from_numpy(scores_sel)
    keep = nms(boxes_t, scores_t, iou_thres).numpy()

    boxes_norm = boxes_norm[keep]
    scores_sel = scores_sel[keep]
    class_ids = class_ids[keep]
    mask_coeffs = mask_coeffs[keep]

    # 保存 NMS 后的 boxes 原始输入尺度，用于 mask
    boxes_input = boxes_norm.copy()

    boxes_norm[:, [0, 2]] -= pad_x
    boxes_norm[:, [1, 3]] -= pad_y
    boxes_orig = boxes_norm / scale

    h0, w0 = orig_shape
    boxes_orig[:, [0, 2]] = np.clip(boxes_orig[:, [0, 2]], 0, w0 - 1)
    boxes_orig[:, [1, 3]] = np.clip(boxes_orig[:, [1, 3]], 0, h0 - 1)

    return boxes_orig, scores_sel, class_ids, proto_out, mask_coeffs, boxes_input

# ------------------------- Mask 生成 -------------------------
def proto_to_masks(proto_out, mask_coeffs, boxes_input, orig_shape, scale, pad_x, pad_y, input_size=640, mask_threshold=0.5):
    p = np.array(proto_out)
    if p.ndim == 4 and p.shape[0] == 1:
        p = p.squeeze(0)
    if p.ndim != 3:
        raise ValueError("proto_out must be shape (proto_c, ph, pw) or (1, proto_c, ph, pw)")

    proto_c, ph, pw = p.shape
    proto_map = np.transpose(p, (1, 2, 0)).astype(np.float32)

    coeffs = np.array(mask_coeffs, dtype=np.float32)
    if coeffs.ndim == 1:
        coeffs = coeffs[np.newaxis, :]
    N, K = coeffs.shape
    use_k = min(K, proto_c)
    proto_map = proto_map[:, :, :use_k]
    coeffs = coeffs[:, :use_k]

    masks_low = np.tensordot(proto_map, coeffs.T, axes=([2], [0]))

    h0, w0 = orig_shape
    masks_out = []

    scaled_w = int(round(w0 * scale))
    scaled_h = int(round(h0 * scale))
    x0 = pad_x
    y0 = pad_y
    x1 = pad_x + scaled_w
    y1 = pad_y + scaled_h

    for i in range(masks_low.shape[2]):
        mask_lr = masks_low[:, :, i]
        mask_input = cv2.resize(mask_lr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        mask_input = 1.0 / (1.0 + np.exp(-np.clip(mask_input, -88.0, 88.0)))
        mask_cropped = mask_input[y0:y1, x0:x1]
        mask_orig = cv2.resize(mask_cropped, (w0, h0), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_orig >= mask_threshold).astype(np.uint8) * 255
        masks_out.append(mask_bin)

    return masks_out

# ------------------------- Mask + Box 可视化 -------------------------
def overlay_masks_on_image(orig_img, masks, boxes=None, labels=None, colors=None, alpha=0.45):
    img = orig_img.copy()
    N = len(masks)
    if colors is None:
        colors = [FIXED_COLORS[i % len(FIXED_COLORS)] for i in range(N)]

    overlay = img.copy()
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        color = colors[i % len(colors)]
        m = (mask > 0).astype(np.uint8)
        if m.sum() == 0:
            continue
        colored = np.zeros_like(img, dtype=np.uint8)
        colored[:, :, 0] = color[0]
        colored[:, :, 1] = color[1]
        colored[:, :, 2] = color[2]
        overlay[m == 1] = cv2.addWeighted(img, 1 - alpha, colored, alpha, 0)[m == 1]

    out = overlay
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            color = colors[i % len(colors)]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if labels is not None:
                label = labels[i]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                cv2.putText(out, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    return out

# ------------------------- 可视化 bbox -------------------------
def visualize_detections(orig_img, boxes, scores, class_ids, class_names, score_thresh=0.3):
    img = orig_img.copy()
    for box, score, cls in zip(boxes, scores, class_ids):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = f"{class_names[int(cls)]} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), (0,255,0), -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    return img

# ------------------------- 主推理 -------------------------
def run_inference(model_path, image_path):
    interpreter = MNN.Interpreter(model_path)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)

    input_size = 640
    img, orig_shape, scale, pad_x, pad_y= preprocess_image(image_path, (input_size, input_size))

    tmp_input = MNN.Tensor(
        (1, 3, input_size, input_size),
        MNN.Halide_Type_Float,
        img.flatten(),
        MNN.Tensor_DimensionType_Caffe
    )
    
    print("img: ", img.flatten()[:10])
    
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    output0 = interpreter.getSessionOutput(session, "output0")
    output1 = interpreter.getSessionOutput(session, "output1")

    # 输出前10个结果
    print("output0:", np.array(output0.getData(), dtype=np.float32)[:10])
    print("output1:", np.array(output1.getData(), dtype=np.float32)[:10])

    output0_data = np.array(output0.getData(), dtype=np.float32).reshape(output0.getShape())
    output1_data = np.array(output1.getData(), dtype=np.float32).reshape(output1.getShape())

    print("DET shape:", output0_data.shape)
    print("PROTO shape:", output1_data.shape)

    boxes, scores, class_ids, proto, mask_coeffs, boxes_input = decode_yolo_seg(
        [output0_data, output1_data],
        orig_shape, scale, pad_x, pad_y, input_size=input_size
    )

    masks = proto_to_masks(proto, mask_coeffs, boxes_input, orig_shape, scale, pad_x, pad_y, input_size=input_size, mask_threshold=0.35)

    coco_names = [
        "person","bicycle","car","motorcycle","airplane","bus","train",
        "truck","boat","traffic light","fire hydrant","stop sign","parking meter",
        "bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
        "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis",
        "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
        "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon",
        "bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog",
        "pizza","donut","cake","chair","couch","pottedplant","bed","diningtable",
        "toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave",
        "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
        "teddy bear","hair drier","toothbrush"
    ]

    origin_img = cv2.imread(image_path)
    labels = [f"{coco_names[int(cid)]} {float(scores[i]):.2f}" for i,cid in enumerate(class_ids)]
    vis_img = overlay_masks_on_image(origin_img, masks, boxes=boxes, labels=labels)
    cv2.imwrite("result_with_masks.jpg", vis_img)

    print(f"Detected {len(boxes)} objects.")
    for i in range(len(boxes)):
        print(f"  Box: {boxes[i]}, Score: {scores[i]:.3f}, Class: {class_ids[i]}")

# ------------------------- 主函数 -------------------------
if __name__ == "__main__":
    model_path = "./yolo11l-seg_int8.mnn"
    image_path = "./bus.jpg"
    run_inference(model_path, image_path)
