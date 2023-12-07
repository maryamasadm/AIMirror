import sys
import os
from fastai.vision.all import *


def predict_bigfive(model_path, user_image_path):
    # Load the model
    
    learn = load_learner(model_path)

    # Load the image
    img = PILImage.create(user_image_path)
    
    # Make a prediction
    _, pred_idx, outputs = learn.predict(img)

    # Extract class name from the vocabulary
    class_name = learn.dls.vocab

    # Print predicted class label and its corresponding probability
    #print(f'Predicted class: {class_name}')
    #print(f'Predicted class: {outputs}')
    for class_name, probability in zip(class_name, outputs):
        print(f'{class_name}: {probability:.2f}')
    class_name = learn.dls.vocab[pred_idx]
    print('the class with higher probebility = ',class_name )

    
    # class names: ['0-Agreeableness', 'Conscientionsness', 'Extraversion', 'Neuritisism', 'Opennes']  


if __name__ == "__main__":
    model_path = 'weights/export.pkl'
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
    #print(image_path)
    # Call the function to process the image
    predict_bigfive(model_path,user_image_path)