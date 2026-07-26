#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 #
# @Time    : 2024/3/28 13:03
# @Author  : # @Email   : # @File    : 测试.py
# @Software: PyCharm

from ultralytics import YOLO
import os
import cv2
import numpy as np

# Path to the model weights
weights_path = r"..\weights\best.pt"

# Directory containing the input images
images_dir = r'..\images'

# Directory to save the prediction results
output_dir = r"..\self"
os.makedirs(output_dir, exist_ok=True)

model = YOLO(weights_path)

for filename in os.listdir(images_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):

        image_path = os.path.join(images_dir, filename)
        print(f"Processing file: {image_path}")

        results = model(image_path, save=False, conf=0.3)

        img = results[0].plot()

        if results[0].masks:
            num_instances = len(results[0].masks.data)
        else:
            num_instances = 0

        text = f"Instances: {num_instances}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        color = (0, 0, 255)
        thickness = 14
        position = (50, 150)

        cv2.putText(img, text, position, font, font_scale, color, thickness)

        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to {output_path}")
