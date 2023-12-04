from ultralytics import YOLO
import cv2
import numpy as np
import dlib
import math

def load_yolo_model():
    try:
        return YOLO("yolov8n.pt")  # Loading the YOLOv8 model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None

def yolo(image, model):
    if model is None:
        return {}
    results = model.predict(source=image, save=True, save_txt=True, conf=0.4)
    names = model.names
    objects_detected = {
        "people_counter": 0,
        "bicycle": 0,
        "dog": 0,
        "boat": 0,
        "cat": 0,
        "skis": 0,
        "tie": 0,
        "wineglass": 0,
        "tennis racket": 0,
        "skateboard": 0,
        "sports ball": 0
    }
    for r in results:
        for c in r.boxes.cls:
            obj_name = names[int(c)]
            if obj_name in objects_detected:
                objects_detected[obj_name] += 1
    return objects_detected

def face_complete_distance_center_light(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_info = {"face_count": len(faces), "face_visibility": "", "distance_to_camera": "", "position": "", "lighting": ""}

    if len(faces) > 0:
        face = faces[0]  # Considering the first detected face
        landmarks = predictor(gray, face)

        # Check face visibility
        visible_landmarks = sum(1 for i in range(68) if landmarks.part(i).y > 0)
        face_info["face_visibility"] = "visible" if visible_landmarks == 68 else "not fully visible"

        # Distance to camera
        face_ratio_to_image_size = (face.width() * face.height()) / (image.shape[1] * image.shape[0])
        if face_ratio_to_image_size <= 0.06:
            face_info["distance_to_camera"] = "too far"
        elif face_ratio_to_image_size >= 0.15:
            face_info["distance_to_camera"] = "too close"
        else:
            face_info["distance_to_camera"] = "fine"

        # Position check
        center_x_threshold = image.shape[1] * 0.5
        upper_y_threshold = image.shape[0] * 0.3
        face_info["position"] = "fine" if face.left() < center_x_threshold and face.top() < upper_y_threshold else "off-center"

        # Lighting condition
        face_roi = gray[face.top():face.top() + face.height(), face.left():face.left() + face.width()]
        average_intensity = cv2.mean(face_roi)[0]
        if average_intensity < 80:
            face_info["lighting"] = "too dark"
        elif 80 <= average_intensity <= 150:
            face_info["lighting"] = "balanced"
        else:
            face_info["lighting"] = "too light"

    return face_info

def glass_detection(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces = detector(image)
    glasses_present = False

    if len(faces) > 0:
        face = faces[0]
        sp = predictor(image, face)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        nose_bridge = landmarks[27:36]  # Nose bridge points
        x_min, x_max = nose_bridge[:, 0].min(), nose_bridge[:, 0].max()
        y_min, y_max = nose_bridge[:, 1].min(), landmarks[31][1]
        
        # Crop and process the image for glass detection
        cropped_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)[y_min:y_max, x_min:x_max]
        img_blur = cv2.GaussianBlur(cropped_img, (3, 3), sigmaX=0, sigmaY=0)
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
        edges_center = edges.T[int(len(edges.T) / 2)]
        count_255 = np.count_nonzero(edges_center == 255)
        count_0 = np.count_nonzero(edges_center == 0)
        glasses_present = count_255 / count_0 >= 0.02

    return glasses_present

def process_image(user_image_path):
    try:
        model = load_yolo_model()
        image = cv2.imread(user_image_path)
        results = {
            "yolo_results": yolo(image, model),
            "face_info": face_complete_distance_center_light(image),
            "glasses_present": glass_detection(image)
        }
        return results
    except Exception as e:
        return {"error": str(e)}

# The __main__ block is removed as it is not necessary for Streamlit integration
