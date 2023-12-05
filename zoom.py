


from PIL import Image
import cv2
import numpy as np
import os
import csv
import sys
import os
from PIL import Image
import dlib
import math
import matplotlib.pyplot as plt
#import streamlit as st
from ultralytics import YOLO


def yolo(image):
    model = YOLO("yolov8n.pt")
    results = model.predict(source=image, save=True, save_txt=True, conf=0.4 )  # save predictions as labels
    #print(len(results[0].boxes.data))
    names = model.names
    people_counter = 0
    bicycle = 0
    dog = 0
    boat = 0
    cat = 0
    skis = 0
    tie = 0
    wineglass = 0
    tennisracket = 0
    skateboard = 0 
    sportsball = 0
    # Add your image processing code, e.g., resizing, filtering, etc.
    #print(f"Image processing completed for: {user_image_path}")
    for r in results:
        for c in r.boxes.cls:
            #print(names[int(c)])
            if names[int(c)] == 'person':
                people_counter +=1
            if names[int(c)] == 'bicycle':
                bicycle ==1
            if names[int(c)] == 'dog':
                dog == 1
            if names[int(c)] == 'boat':
                boat == 1
            if names[int(c)] == 'cat':
                cat == 1         
            if names[int(c)] == 'skis':
                skis == 1 
            if names[int(c)] == 'tie':
                tie == 1 
            if names[int(c)] == 'wine glass':
                wineglass == 1 
            if names[int(c)] == 'tennis racket':
                tennisracket == 1 
            if names[int(c)] == 'skateboard':
                skateboard == 1 
            if names[int(c)] == 'sports ball':
                sportsball == 1 
    print('people_counter = ',people_counter)
    print('bicycle = ',bicycle)
    print('dog = ',dog)
    print('boat = ',boat)
    print('cat = ',cat)
    print('skis = ',skis)
    print('tie = ',tie)
    print('wine glass = ',wineglass)
    print('tennis racket = ',tennisracket)
    print('skateboard = ',skateboard)
    print('sports ball = ',sportsball)
    return people_counter

def  face_complete_distance_center_light(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = detector(gray)
    all_landmarks = []
    #print('len(faces) = ', len(faces))
    # Check if any faces are detected
    if len(faces) > 0:
        # Loop through each face and get facial keypoints
        for face in faces:
            landmarks = predictor(gray, face)

            # Draw facial keypoints on the image
            for i in range(68):  # 68 landmarks for a face
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                all_landmarks.append((x, y))
            # Count the number of visible landmarks (modify the condition based on your criteria)
        visible_landmarks = sum(1 for x, y in all_landmarks if y > 0)
        #print(f"Number of visible landmarks: {visible_landmarks}")

        # next stage for checking face is visible including forehead
        
        if visible_landmarks ==68:
            # Get the bounding box coordinates of the detected face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # Calculate the length of the nose (distance between landmarks 27 and 30)
            nose_length = math.sqrt((landmarks.part(30).x - landmarks.part(27).x)**2 + (landmarks.part(30).y - landmarks.part(27).y)**2)
            if y >= nose_length:
                print("The face is visible")
            else:
                print("the face is not completely visible")


            # compute if the person is too close to the camera : distance            
            # Get the bounding box coordinates of the detected face
            face_x, face_y, face_w, face_h = face.left(), face.top(), face.width(), face.height()

            # Get the bounding box coordinates of the entire image
            image_h, image_w, _ = image.shape

            # Calculate the ratio of the face bounding box to the entire image bounding box
            face_ratio_to_imagesize = (face_w * face_h) / (image_w * image_h)
            #print('face_ratio_to_imagesize = ',face_ratio_to_imagesize)
            if face_ratio_to_imagesize <= 0.06:
                print('you are too far, please come closer !')
            elif  face_ratio_to_imagesize >= 0.15:
                print('you are too close, please keep more distance !')
            else:
                print('the distance to camera is fine!')
            # Display the result
            '''cv2.imshow("Facial Keypoints Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''

            # compute if the person sit in the center
            image_height, image_width, _ = image.shape
            center_threshold = 0.5  # Adjust this value as needed
            upper_threshold = 0.3   # Adjust this value as needed

            # Calculate the center and upper part thresholds
            center_x_threshold = image_width * center_threshold
            upper_y_threshold = image_height * upper_threshold

            # Check if the face bounding box is in the center and upper part of the image
            if x < center_x_threshold and y < upper_y_threshold:
                print("the position is fine")
            else:
                print("please sit at the center of the screen")
           

            # Compute the average pixel intensity within the face ROI: lightening
            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Extract the region of interest (ROI) corresponding to the face
            face_roi = gray_image[y:y+h, x:x+w]
            average_intensity = cv2.mean(face_roi)[0]
            #print('average_intensity =', average_intensity)
            # Determine the lighting condition within the face ROI
            print('average_intensity= ', average_intensity)
            if average_intensity < 80:
                print('Too Dark, please turn on a light')
            elif 80 <= average_intensity <= 150:
                print('lightening is balanced')
            else:
                print('Too light, please turn off a light')

            # Compute the Laplacian of the image
            laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)

            # Calculate the variance of the Laplacian
            variance = laplacian.var()
            print(variance)
            # Determine if the image is blurry based on the variance
            if variance > 800:
                print('it seems the camera is not clean ! a bit blur')
            if variance < 800:
                print('camera is clean, blurness is not found !')
            #return is_blurry, variance



    return len(faces)

def  glass_detection(image, user_image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    rect = detector(image)[0]
    sp = predictor(image, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    nose_bridge_x = []
    nose_bridge_y = []
    for i in [28,29,30,31,33,34,35]:
            nose_bridge_x.append(landmarks[i][0])
            nose_bridge_y.append(landmarks[i][1])      
    ### x_min and x_max
    x_min = min(nose_bridge_x)
    x_max = max(nose_bridge_x)
    ### ymin (from top eyebrow coordinate),  ymax
    y_min = landmarks[20][1]
    y_max = landmarks[31][1]
    img2 = Image.open(user_image_path)
    img2 = img2.crop((x_min,y_min,x_max,y_max))
    img_blur = cv2.GaussianBlur(np.array(img2),(3,3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image =img_blur, threshold1=100, threshold2=200)
    #plt.imshow(edges, cmap =plt.get_cmap('gray'))
    #center strip
    edges_center = edges.T[(int(len(edges.T)/2))]
    count_255 = np.count_nonzero(edges_center == 255)
    count_0 = np.count_nonzero(edges_center == 0)
    #print('number of 255',count_255/count_0)
    if count_255/count_0 >= 0.02:
        print('Glasses are present')
    else:
        print('Glasses are not present')


def process_image(user_image_path):
    # Add your image processing logic here
    try:
        image = cv2.imread(user_image_path)
        # YOLO
        num_person = yolo(image)
        num_face = face_complete_distance_center_light(image)
        if num_face == 0 and num_person != 0:
            print('the face is not completely visible, too close or too far')
        if num_face != 0 and num_person != 0:
            glass_detection(image,user_image_path )

    except Exception as e:
        print(f"Error processing image: {e}")
   





if __name__ == "__main__":
    # Check if the user provided an image path as a command-line argument
    if len(sys.argv) != 2:
        print("Usage: python process_images.py <image_path>")
        sys.exit(1)

    # Get the image path from the command-line argument
    user_image_path = sys.argv[1]

    # Check if the entered path is valid
    if not os.path.isfile(user_image_path):
        print("Invalid path. Please make sure the file exists.")
        sys.exit(1)

    # Call the function to process the image
    process_image(user_image_path)






        
print("Processing complete.")
