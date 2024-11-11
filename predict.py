from ultralytics import YOLO

model = YOLO('plant_id_v2.pt')

path = 'D:\School\SP\plants\V2'

save_path='..\\radai\\outputs'

results = model(source=path, conf=0.25, save=True, save_dir=save_path)