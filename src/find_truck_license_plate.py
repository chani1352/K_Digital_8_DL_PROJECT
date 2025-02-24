import torch
import yolov5
import easyocr
import numpy as np
import cv2
from PIL import Image

def load_model_license_plate(device):
     model = yolov5.load('keremberke/yolov5m-license-plate')
     model.conf = 0.25  # 신뢰도 기준
     model.iou = 0.45  # 박스 겹침 정도
     model.agnostic = False  # 객체 종류별로 중복 처리
     model.multi_label = False  # 하나의 객체에 여러 라벨 방지
     model.max_det = 1000  # 최대 탐지 객체 수
     model.to(device)
     return model

def find_truck_license_plate(img, model):
    if model is None:
        raise ValueError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")
    try:
        reader = easyocr.Reader(['ko'], gpu=True) 
        result = model(img, size=640, augment=True)
        result_img = result.ims[0]
        result_xywh = torch.tensor(result.xywh[0])
        license_plate_class_id = 0 
        filtered_indices = result_xywh[result_xywh[:, 5] == license_plate_class_id]
        for detection in filtered_indices:
            x_center, y_center, width, height, confidence, class_id = detection
            x_min = int((x_center - width / 2))
            y_min = int((y_center - height / 2))
            x_max = int((x_center + width / 2))
            y_max = int((y_center + height / 2))
            license_plate_roi = result_img[y_min:y_max, x_min:x_max]
            results = reader.readtext(license_plate_roi) 
            if results:
                return license_plate_roi
        return None  
    except Exception as e:
        print(f"번호판 영역 추출 중 오류 발생: {e}")
        return None

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model_license_plate(device)
    image_path = "input.jpg"  
    image = Image.open(image_path)
    license_plate = find_truck_license_plate(image, model)
    if license_plate is not None:
        cv2.imwrite("output.jpg", license_plate) 