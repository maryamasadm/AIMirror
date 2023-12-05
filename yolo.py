from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco128.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("/Users/maryam/Desktop/DS/A/9136232.jpeg")  # predict on an image
print(results)
path = model.export(format="onnx")  # export the model to ONNX format