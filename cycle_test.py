
# resnet循环测试ROCm     
# windows下如果想加入iGPU的测试，请在其他环境下载onnxruntime-directml测试 , 并设置providers=["DmlExecutionProvider"] 
import os 
os.environ['HSA_OVERRIDE_GFX_VERSION'] = "11.0.0"
import cv2
import numpy as np
import onnxruntime as ort
from matplotlib import pyplot as plt
import time
print(f" [INFO] Resnet50 cycle tested on the CPU and NPU of the HX370")


#模型的ExecutionProvider设置
model = "./toturials_models/resnet50/ResNet_int.onnx"

# session_npu = ort.InferenceSession(
#             model,
#             providers=["VitisAIExecutionProvider"],
#             provider_options=provider_options
#         )
session_rocm = ort.InferenceSession(
            model,
            providers=["ROCMExecutionProvider"],
        )

def preprocess_image(image_path):
    # 加载图像并调整大小
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB
    img = cv2.resize(rgb_img, (224, 224))  # 调整为 224x224
    
    # 转换为 float32 类型，并归一化到 [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # 标准化
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    
    # 调整通道顺序为 [C, H, W]，再扩展为 [1, C, H, W],匹配模型输入
    #img = np.transpose(img, (2, 0, 1))  # 转换为 [C, H, W]
    img = np.expand_dims(img, axis=0)  # 转换为 [1, C, H, W]
    
    return rgb_img, img

# 加载并预处理图像
# image_path = r"AI_story\temp\person.jpg"  # 测试图像路径
image_path = "AI_story/temp/person.jpg"
rgb_img, input_tensor = preprocess_image(image_path)

# 获取模型输入名称
input_name = session_rocm.get_inputs()[0].name

start = time.time()
iter = 1000

for i in range(iter):
    output = session_rocm.run(None, {input_name: input_tensor})[0]
end =time.time()

# for i in range(iter):
#     output = session.run(None, {input_name: input_tensor})[0]
# end_cpu =time.time()


# print(f" [INFO]  Run {iter} iters on HX370 CPU using {end_cpu-start_cpu:.4f} seconds")
print(f" [INFO]  Run {iter} iters on HX370 NPU using {end-start:.4f} seconds")



