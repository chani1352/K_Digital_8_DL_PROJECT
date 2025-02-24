import torch  
import cv2  
import numpy as np  
from src.textrecog import ImageClassifier
digit_weights_path = 'weights/persian_digit_classifier.pt'
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','영']
classifier = ImageClassifier(digit_weights_path, class_names)
yolo_weight_path='weights/yolov5.pt'

def ensure_grayscale(img):
    if len(img.shape) == 3:  
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    return img 

def process_yolo(img):
    pix_min = []
    plate_number = ''
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_weight_path)  
    model.conf = 0.5
    model.iou = 0.3  # 박스 겹침 정도
    model.agnostic = True  # 객체 종류별로 중복 처리
    model.multi_label = False  # 하나의 객체에 여러 라벨 방지
    model.max_det = 1000  # 최대 탐지 객체 수
    model.eval()
    img = np.array(img)  
    img_recognition = img.copy()
    results = model(img)  
    boxes = results.xywh[0][:, :4].cpu().numpy()  

    #conf = results.xywh[0][:, 4].cpu().numpy()  
    #class_ids = results.xywh[0][:, 5].cpu().numpy().astype(int)  # 클래스 ID 추출
    #labels = [results.names[int(c)] for c in class_ids]  # 클래스 이름 추출
    
    if boxes.shape[0] == 0:
        print("No objects detected.")
        return img, img
    
    boxes = boxes[boxes[:, 0].argsort()]
    xTL, yTL = boxes[0, 0], boxes[0, 1]
    xBR, yBR = boxes[-1, 0], boxes[-1, 1]

    if (xBR - xTL) == 0:
        m = 0 
        b = yTL
    else:
        m = (yBR - yTL) / (xBR - xTL)  
        b = yTL - m * xTL 

    top_row_imgs = [] 
    bottom_row_imgs = [] 

    for i in range(len(boxes)):
        x_center, y_center, width, height = boxes[i]
        x_min, y_min = int(x_center - width / 2), int(y_center - height / 2)
        x_max, y_max = int(x_center + width / 2), int(y_center + height / 2)
        cv2.rectangle(img_recognition, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        # 신뢰도 및 클래스 라벨 추가
        # label = f"{class_ids[i]}: {conf[i]:.2f}"  # 클래스 이름과 신뢰도를 표시 (소수점 2자리까지)
        # cv2.putText(image_box, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if m == 0:
            if b < y_max:
                crossing = True
            else:
                crossing = False
        else:
            y_line = m * x_center + b
            if y_line < y_max:
                crossing = True
            else:
                crossing = False

        cropped_img = img[y_min:y_max, x_min:x_max]
        cropped_img = ensure_grayscale(cropped_img)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cropped_img)
        pix_min.append(min_val)
        cropped_img = cv2.resize(cropped_img,(150,300))
        if crossing :
            bottom_row_imgs.append(cropped_img)
        else :
            top_row_imgs.append(cropped_img)
    cv2.line(img_recognition, (int(xTL),int(yTL)), (int(xBR),int(yBR)), (0, 0, 255), 5)

    if top_row_imgs is not None:
        allImage = top_row_imgs
        allImage.extend(bottom_row_imgs)
    else:
        allImage = bottom_row_imgs 
    min_mean = np.mean(pix_min)
    box_img = []
    for img in allImage:
        img_avg = np.mean(img) + 5
        if min_mean < 60 :
            ret, img = cv2.threshold(img, img_avg, 255, cv2.THRESH_BINARY_INV)
        else :
            ret, img = cv2.threshold(img, img_avg, 255, cv2.THRESH_BINARY)
        img = cv2.copyMakeBorder(img, 30, 30, 20, 20, cv2.BORDER_CONSTANT, value=0)  
        if classifier.predict(img) == '영':
            continue
        plate_number += classifier.predict(img)
        box_img.append(img)
    combined_img = box_img[0]
            
    for img in box_img[1:]:
        combined_img = np.hstack([combined_img, img])

    return img_recognition, combined_img, plate_number


if __name__ == "__main__":
    img_path = "test_image.jpg"  
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"이미지 파일 '{img_path}'을 찾을 수 없습니다.")

        img_recognition, combined_img, plate_number = process_yolo(img)

        if plate_number:
            print("Recognized Plate Number:", plate_number)
            cv2.imshow("Recognition Result", img_recognition)
            cv2.imshow("Combined Characters", combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No plate number recognized.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")