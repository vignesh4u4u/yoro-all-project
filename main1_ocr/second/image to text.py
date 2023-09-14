from flask import Flask, request, jsonify, json
import cv2
import numpy as np
from paddleocr import PaddleOCR
app = Flask(__name__)
ocr = PaddleOCR(use_angle_cls=True, lang='en')
@app.route("/ml-service/image-text/v1/ping", methods=["GET"])
def home():
    if request.method == 'GET':
        return "pong"
@app.route("/ml-service/image-text", methods=["POST"])
def image_text_conversion():
    if request.method == "POST":
        file = request.files['file']
        file_end = file.filename.endswith(".pdf")
        if file_end == True:
            return jsonify({"error": "PDF file detected, not supported yet"})
        else:
            image_data = file.read()
            gray = cv2.cvtColor(cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
            result = ocr.ocr(gray)
            detected_text_list = []
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    detected_text_list.append(line[1][0])
            text = ' '.join(detected_text_list)
            return jsonify({"text": text})
        #http://localhost:5000/ml-service/image-text
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)