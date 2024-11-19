from ultralytics import YOLO

model = YOLO('best171124.pt')

path = '.\\kidney_stone.jpg'

save_path='..\\outputs\\'

results = model(source=path, conf=0.25, save=True, save_dir=save_path)