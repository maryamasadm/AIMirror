

from ultralytics import YOLO
from PIL import Image
import cv2
import cv2
import numpy as np
import os
#from os import listdir
import csv

#model = YOLO("model.pt")
model = YOLO("yolov8n.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
#results = model.predict(source="0")


def save_to_csv(csv_file, image_name, people_counter, bicycle, dog, boat,cat, skis,tie,suitcase,wineglass, tennisracket, skateboard, sportsball):
    with open(csv_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if people_counter is not None:
            #csv_writer.writerow([image_name] + result.flatten().tolist())
            csv_writer.writerow([image_name] + [people_counter] + [bicycle] + [dog] + [boat] + [cat] + [skis] + [tie]+ [suitcase] + [wineglass]+ [tennisracket]  + [skateboard] + [sportsball])


csv_file = 'results.csv'
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['ImageName'] + ['person']+ ['bicycle'] + ['dog'] + ['boat'] + ['cat'] + ['skis'] + ['tie']+ ['suitcase'] + ['wineglass'] + ['tennis racket']  + ['skateboard'] + ['sportsball'])  # Add column headers




# Initialize rounded_prediction variable before the loop
rounded_prediction = None
# get the path or directory
folder_dir = "/Users/maryam/Desktop/DS/Neuroticism/"
for images in os.listdir(folder_dir):

    # check if the image ends with png or jpg or jpeg
    if (images.endswith(".png") or images.endswith(".JPG") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        # display
        print(images)
        image = cv2.imread(os.path.join("/Users/maryam/Desktop/DS/Neuroticism/", images))
        results = model.predict(source=image, save=True, save_txt=True, conf=0.4 )  # save predictions as labels
        print(len(results[0].boxes.data))
        names = model.names
        people_counter = 0
        bicycle = 0
        dog = 0
        boat = 0
        cat = 0
        skis = 0
        tie = 0
        suitcase = 0
        wineglass = 0
        tennisracket = 0
        skateboard = 0 
        sportsball = 0

        for r in results:
            for c in r.boxes.cls:
                print(names[int(c)])
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
                if names[int(c)] == 'suitcase':
                    suitcase == 1 
                if names[int(c)] == 'wine glass':
                    wineglass == 1 
                if names[int(c)] == 'tennis racket':
                    tennisracket == 1 
                if names[int(c)] == 'skateboard':
                    skateboard == 1 
                if names[int(c)] == 'sports ball':
                    sportsball == 1 
        print('people_counter = ',people_counter)
        save_to_csv(csv_file, images, people_counter, bicycle, dog, boat,cat, skis,tie,suitcase,wineglass, tennisracket,skateboard, sportsball)

print("Processing and CSV writing complete.")
