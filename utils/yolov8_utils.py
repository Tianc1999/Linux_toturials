import numpy as np
import cv2
from PIL import Image

# 检测类名称和对应的随机颜色
class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))

# 非极大值抑制（NMS）
def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]
    return keep_boxes

# 多类别 NMS
def multiclass_nms(boxes, scores, class_ids, iou_threshold):
    unique_class_ids = np.unique(class_ids)
    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices, :]
        class_scores = scores[class_indices]
        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])
    return keep_boxes

# 计算 IoU
def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    return intersection_area / union_area

# 转换边界框格式
def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# 绘制检测结果
def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)
    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]
        draw_box(det_img, box, color)
        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)
    return det_img

def draw_box(image, box, color=(0, 0, 255), thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

def draw_text(image, text, box, color=(0, 0, 255), font_size=0.001, text_thickness=2):
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)
    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)
    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_masks(image, boxes, classes, mask_alpha=0.3):
    mask_img = image.copy()
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)


def process_output_and_save(output, image, conf_threshold, iou_threshold, save_path):
    predictions = np.squeeze(output).T

    # 获取分数和类别
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_threshold, :]
    scores = scores[scores > conf_threshold]

    if len(scores) == 0:
        print("No objects detected.")
        return

    # 获取类别 ID 和边界框
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = xywh2xyxy(predictions[:, :4])

    # NMS 处理
    keep_indices = multiclass_nms(boxes, scores, class_ids, iou_threshold)
    boxes, scores, class_ids = boxes[keep_indices], scores[keep_indices], class_ids[keep_indices]

    # 绘制结果并保存
    result_image = draw_detections(image, boxes, scores, class_ids)
    cv2.imwrite(save_path, result_image)
    print(f"Result saved at: {save_path}")


def get_labels_from_txt(path):
    labels_dict = dict()
    with open(path) as f:
        for cat_id, label in enumerate(f.readlines()):
            labels_dict[cat_id] = label.strip()
    return labels_dict


def v8_nms(pred, conf_thres, iou_thres): 
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True] 
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))  
    output_box = []  
    for i in range(len(total_cls)):
        clss = total_cls[i] 
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]  
        box_conf_sort = np.argsort(box_conf) 
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box) 
        cls_box = np.delete(cls_box, 0, 0) 
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]  
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]  
                interArea = v8_getInter(max_conf_box, current_box)  
                iou = v8_getIou(max_conf_box, current_box, interArea)  
                if iou > iou_thres:
                    del_index.append(j)  
            cls_box = np.delete(cls_box, del_index, 0)  
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box
 
########################yolov8 func###############################
def v8_getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou
 
 
def v8_getInter(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                         box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box1[3] / 2, \
                                         box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter
 

def v8_draw(img, xscale, yscale, pred, names, color=(0, 255, 0), wt=1):
    img_ = img.copy()
    if len(pred):
        for detect in pred:
            # 获取检测框的坐标并缩放到原图尺寸
            x1, y1, x2, y2 = (int((detect[0] - detect[2] / 2) * xscale), int((detect[1] - detect[3] / 2) * yscale),
                              int((detect[0] + detect[2] / 2) * xscale), int((detect[1] + detect[3] / 2) * yscale))
            class_id = int(detect[5])
            conf = detect[4]

            # 绘制检测框
            img_ = cv2.rectangle(img_, (x1, y1), (x2, y2), color, wt)

            # 绘制标签和置信度
            label = f"{names[class_id]} {conf:.2f}"
            img_ = cv2.putText(img_, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img_
########################yolov8 func###############################