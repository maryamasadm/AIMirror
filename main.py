from functions.zoom import process_image
from functions.bigfive import predict_bigfive
import sys
import os

def main(bigfive_model_path, landmarkface_model_path, user_image_path):
    process_image(landmarkface_model_path,yolo_model_path, user_image_path)
    predict_bigfive(bigfive_model_path, user_image_path)

if __name__ == "__main__":
    bigfive_model_path = 'weights/export.pkl'
    landmarkface_model_path = 'weights/shape_predictor_68_face_landmarks.dat'
    yolo_model_path = 'weight/yolov8n.pt'
    
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
    main(bigfive_model_path, landmarkface_model_path, user_image_path)