from flask import Flask,request,render_template,jsonify,json
from paddleocr import PaddleOCR,draw_ocr
import pytesseract as pytes
import numpy as np
import pandas as pd
import seaborn as sns
import pypdfium2 as pdfium
import matplotlib.pyplot as plt
import easyocr
import os
import re
import cv2
from PIL import Image
import requests
import json
ocr = PaddleOCR(use_angle_cls=True,lang='en')
app=Flask(__name__)
@app.route("/ml-service/pdfocr/v1/ping",methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/ocr", methods=["POST"])
def extract_text_information_pdf():
    if request.method == 'POST':
        file = request.files["file"]
        fields_input = request.form["fields"]
        file_end = file.filename.endswith(".pdf")
        #print(file_end)
        if file_end == True:
            file_path = "temp.pdf"
            file.save(file_path)
            script_directory = os.path.dirname(os.path.abspath(__file__))
            temp_images_subdirectory = "temp_images"
            output_dir = os.path.join(script_directory, temp_images_subdirectory)
            os.makedirs(output_dir, exist_ok=True)
            pdf = pdfium.PdfDocument(file_path)
            n_pages = len(pdf)
            image_path_list = []
            os.chdir(output_dir)
            for page_number in range(n_pages):
                page = pdf.get_page(page_number)
                pil_image = page.render(scale=300/72).to_pil()
                image_path = f"image_{page_number + 1}.png"
                pil_image.save(image_path)
                image_path_list.append(image_path)
            detected_text_list = []
            for image_path in image_path_list:
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                result = ocr.ocr(gray)
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        detected_text_list.append(line[1][0])
                text = ' '.join(detected_text_list)
                #print(text)
                data = {}
                fields = json.loads(fields_input)
                if fields_input:
                    for field in fields:
                        key = field.get("key")
                        pattern = field.get("pattern")
                        repeatable = field.get("repeatable", True)
                        table = field.get("table")
                        if table == False:
                            if pattern:
                                matches = re.findall(pattern, text, flags=re.IGNORECASE)
                                if matches and table == False:
                                    if repeatable and table == False:
                                        data[key] = matches
                                    else:
                                        data[key] = matches[0]
                        if table == True:
                            table_pattern = json.loads(fields_input)
                            matched_data = {}
                            for f in table_pattern:
                                key1 = f.get("key")
                                pattern1 = f.get("pattern")
                                repeatable1 = f.get("repeatable", True)
                                table1 = f.get("table")
                                if table1 == True:
                                    if pattern1:
                                        matches1 = re.findall(pattern1, text, flags=re.IGNORECASE)
                                        if matches1 and table1 == True:
                                            matched_data[key1] = matches1
                            output_data = []
                            keys = list(matched_data.keys())
                            if keys:
                                max_entries = max(len(matched_data[key]) for key in keys)
                                for i in range(max_entries):
                                    entry = {}
                                    for key in keys:
                                        if i < len(matched_data[key]):
                                            entry[key] = matched_data[key][i]
                                    output_data.append(entry)
                                data['table_data'] = output_data
                            else:
                                data['table_data'] = "No matching data found for table patterns."
                if data:
                    return jsonify(data)
                else:
                    return jsonify({"error": "No matching data found"})
            os.remove(file_path)
        else:
            image_data = file.read()
            fields = json.loads(fields_input)
            gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            result = ocr.ocr(gray)
            detected_text_list = []
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    detected_text_list.append(line[1][0])
            text = ' '.join(detected_text_list)
            data = {}
            if fields_input:
                for field in fields:
                    key = field.get("key")
                    pattern = field.get("pattern")
                    repeatable = field.get("repeatable", True)
                    table = field.get("table")
                    if table == False:
                        if pattern:
                            matches = re.findall(pattern, text, flags=re.IGNORECASE)
                            if matches and table == False:
                                if repeatable and table == False:
                                    data[key] = matches
                                else:
                                    data[key] = matches[0]
                    if table == True:
                        table_pattern = json.loads(fields_input)
                        matched_data = {}
                        for f in table_pattern:
                            key1 = f.get("key")
                            pattern1 = f.get("pattern")
                            repeatable1 = f.get("repeatable", True)
                            table1 = f.get("table")
                            if table1 == True:
                                if pattern1:
                                    matches1 = re.findall(pattern1, text, flags=re.IGNORECASE)
                                    if matches1 and table1 == True:
                                        matched_data[key1] = matches1
                        output_data = []
                        keys = list(matched_data.keys())
                        if keys:
                            max_entries = max(len(matched_data[key]) for key in keys)
                            for i in range(max_entries):
                                entry = {}
                                for key in keys:
                                    if i < len(matched_data[key]):
                                        entry[key] = matched_data[key][i]
                                output_data.append(entry)
                            data['table_data'] = output_data
                        else:
                            data['table_data'] = "No matching data found for table patterns."
            if data:
                return jsonify(data)
            else:
                return jsonify({"error": "No matching data found"})
    #http://localhost:5000/ml-service/pdfocr
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)