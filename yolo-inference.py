from ultralytics import YOLO

# writing model name
model = YOLO('yolov8s')
results = model.predict('input-videos/08fd33_4.mp4',
                        save=True)  # saving the result

print(results[0])  # printing the result of the first frame

print('--------------------------------')
for box in results[0].boxes:
    print(box)
