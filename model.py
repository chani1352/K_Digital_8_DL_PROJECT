import io
import torch
import numpy as np
import base64
import cv2
import logging

from flask import Flask, request, jsonify, render_template
from PIL import Image

from src.levenshtein import levenshtein_similarity
from src.yolov import process_yolo
from src.find_truck_license_plate import (
    load_model_license_plate,
    find_truck_license_plate,
)

app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_findLP = load_model_license_plate(device)

logging.basicConfig(level=logging.INFO)

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}  # 허용 이미지 확장자 설정
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB 제한


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS
    )



@app.route("/", methods=["GET"])
def upload_page():
    return render_template("upload.html")  # upload.html 템플릿 렌더링

@app.route("/process_image", methods=["POST"])
def process_image():
    try:
        if request.method == "POST":
            if "file" not in request.files:
                return jsonify({"error": "No file part"}), 400

            file = request.files["file"]

            if file.filename == "":
                return jsonify({"error": "No selected file"}), 400

            if not allowed_file(file.filename):
                return (
                    jsonify(
                        {
                            "error": "Invalid file type. Allowed types are: "
                            + ", ".join(ALLOWED_IMAGE_EXTENSIONS)
                        }
                    ),
                    400,
                )

            if len(file.read()) > MAX_FILE_SIZE:
                file.seek(0)
                return (
                    jsonify(
                        {
                            "error": f"File size too large. Maximum allowed size is {MAX_FILE_SIZE // (1024 * 1024)} MB"
                        }
                    ),
                    400,
                )

            file.seek(0)

            img_data = file.read()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            gamma = 0.8
            inv_gamma = 1.0 / gamma
            lut = np.array([((j / 255.0) ** inv_gamma) * 255 for j in range(256)]).astype('uint8')

            license_plate_detection = find_truck_license_plate(img, model_findLP)
            license_plate_detection = np.array(license_plate_detection)
            license_plate_detection = cv2.LUT(license_plate_detection, lut)
            processing, processing_box, result = process_yolo(license_plate_detection)
            number_predict = levenshtein_similarity(result)

            pil_image_processing = Image.fromarray(processing.astype("uint8"))
            pil_image_box = Image.fromarray(processing_box.astype("uint8"))

            img_byte_arr = io.BytesIO()
            pil_image_processing.save(img_byte_arr, format="JPEG")
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

            img_byte_arr1 = io.BytesIO()
            pil_image_box.save(img_byte_arr1, format="JPEG")
            img_base64_1 = base64.b64encode(img_byte_arr1.getvalue()).decode("utf-8")

            #number_predict = result

            return jsonify(
                {
                    "inImg2": img_base64,
                    "inImg3": img_base64_1,
                    "carNumber": number_predict,
                }
            )
    except Exception as e:
        app.logger.error("Error processing image: %s", e, exc_info=True)
        return (
            jsonify(
                {
                    "error": "An error occurred during image processing. Please check the server logs."
                }
            ),
            500,
        )
    
if __name__ == '__main__':
    app.run(host='00.000.000.000', port=5000, debug=True)