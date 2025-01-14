# Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.


####################################################
# npu test
###################################################



import os
import sys
import subprocess
import numpy as np
import onnxruntime as ort
import cv2
import time


model = r"onnx_utils\DetectionModel_int.onnx"
# config_file_path = r"onnx_utils\vaip_config.json"
# cache_key   = 'modelcachekey_yolov8'
# provider_options = [{
#             'config_file': config_file_path,
#             'cacheKey': cache_key,
#         }]

# session_npu = ort.InferenceSession(
#             model,
#             providers=["VitisAIExecutionProvider"],
#             provider_options=provider_options
#         )


# session_cpu = ort.InferenceSession(model)

session_igpu = ort.InferenceSession(model,providers=["DmlExecutionProvider"])

image = r"temp\person.jpg"
img = cv2.imread(image)
input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_img = cv2.resize(input_img, (640, 640))
input_img = input_img / 255.0
input_img = input_img.transpose(2, 0, 1).astype(np.float32)
input_img = np.expand_dims(input_img, axis=0)

input = session_igpu.get_inputs()[0].name
output = session_igpu.get_outputs()[0].name

iter=500

# start_cpu = time.time()
# for i in range(iter):
#     output = session_cpu.run(None,{input:input_img})[0]
# end_cpu =time.time()

start_igpu = time.time()
for i in range(iter):
    output = session_igpu.run(None,{input:input_img})[0]
end_igpu =time.time()


# start_npu = time.time()
# for i in range(iter):
#     result = session_npu.run(None,{input:input_img})[0]
# end_npu = time.time()






# print(f" [INFO]  Yolov8 Run {iter} iters on HX370 CPU using {end_cpu-start_cpu:.4f} seconds")
# print(f" [INFO]  Yolov8 Run {iter} iters on HX370 iGPU using {end_igpu-start_igpu:.4f} seconds")
print(f" [INFO]  Yolov8 Run {iter} iters on HX370 iGPU using {end_igpu-start_igpu:.4f} seconds")

