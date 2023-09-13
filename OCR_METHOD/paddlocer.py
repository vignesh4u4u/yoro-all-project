from flask import Flask, request, send_file,json,jsonify
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
import torch
import pypdfium2 as pdfium
import os
import io
from PIL import Image
import json
ocr = PaddleOCR(use_angle_cls=True, lang='en')
img_path = r'C:\Users\VigneshSubramani\Pictures\invoice image\sample13.png'
result = ocr.ocr(img_path)
detected_text_list = []
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        detected_text_list.append(line[1][0])
text = ' '.join(detected_text_list)
print(text)