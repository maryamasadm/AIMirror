
from fastai.vision.all import *
def predict_image(model_path, image_path):
    # Load the model
    learn = load_learner(model_path)

    # Load the image
    img = PILImage.create(image_path)

    # Make a prediction
    _, pred_idx, outputs = learn.predict(img)

    # Extract class name from the vocabulary
    class_name = learn.dls.vocab
    print(class_name)


    # Print predicted class label and its corresponding probability
    print(f'Predicted class: {class_name}')
    print(f'Predicted class: {outputs}')
    for class_name, probability in zip(class_name, outputs):
        print(f'{class_name}: {probability:.2f}')
    class_name = learn.dls.vocab[pred_idx]
    print('the class with higher probebility = ',class_name )


if __name__ == "__main__":
    model_path = '/Users/maryam/Desktop/export.pkl'
    image_path = '/Users/maryam/Desktop/e.png'

    predict_image(model_path, image_path)
    
    # class names: ['0-Agreeableness', 'Conscientionsness', 'Extraversion', 'Neuritisism', 'Opennes']  