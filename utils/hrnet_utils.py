import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

INPUT_SIZE = [512, 1024]


def input_transform(image):
    """Preprocess an image

    Args:
        img (ndarray): Image to be normalized.

    Returns:
        ndarray: The normalized image.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) 
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def pad_image(image, h, w, size, padvalue):
    pad_image = image.copy()
    pad_h = max(size[0] - h, 0)
    pad_w = max(size[1] - w, 0)
    if pad_h > 0 or pad_w > 0:
        pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                       pad_w, cv2.BORDER_CONSTANT,
                                       value=padvalue)
    return pad_image, pad_h, pad_w


def resize_image(image, re_size, keep_ratio=True):
    if not keep_ratio:
        re_image = cv2.resize(image,                                              
                           (re_size[0], re_size[1])).astype('float32')
        return re_image, 0, 0 
    ratio = re_size[0] * 1.0 / re_size[1] 
    h, w = image.shape[0:2]
    if h * 1.0 / w <= ratio:
        re_h, re_w = int(h * re_size[1] * 1.0 / w), re_size[1] 
    else:
        re_h, re_w = re_size[0], int(w * re_size[0] * 1.0 / h)
    
    re_image = cv2.resize(image,                                               
                          (re_w, re_h)).astype('float32')
    # print(f're_image shape:{re_image.shape}')
    re_image, pad_h, pad_w = pad_image(re_image, re_h, re_w, re_size, (0.0, 0.0, 0.0))
    # print(f're_h: {re_h}, re_w: {re_w}')
    return re_image, pad_h, pad_w


def preprocess(img):
    """Preprocess an image

    Args:
        img (ndarray): Image to be normalized.

    Returns:
        ndarray: The normalized image.
    """
    img, pad_h, pad_w = resize_image(img, INPUT_SIZE)
    img = input_transform(img)

    return img.transpose((2, 0, 1)), pad_h, pad_w 


def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix
