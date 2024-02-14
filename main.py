from ultralytics import YOLO

# Load a model
model = YOLO("bottledetection.pt")  # build a new model from scratch



# Use the model
results = model(source=1, show=True, conf=0.3, save=True )  # train the model