import torch
from ultralytics import YOLO, checks, hub

if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.__version__)
    print("Running")

    checks()

    hub.login('3b5056ac3a9ea918ac838037d777446ba97e9ad3fc')
    

    # Load YOLOv8 model (choose your version)
    try:
        model = YOLO('https://hub.ultralytics.com/models/GeWkX1LhQyAovcnN6bXI')
        # model = YOLO('yolov8n.pt') 

        # Train the model with your dataset
        model.train(
            data='kidney-stone-ultrasound-1\\data.yaml',  # Path to dataset YAML
            epochs=20,  # Start with 20 epochs
            imgsz=320,  # Image size
            plots=True,
            batch=2
        )
    except RuntimeError as e:
        print(f"Caught a RuntimeError: {e}")
        print("It seems there was a CUDA-related issue. Freeing up memory and continuing.")

    except Exception as e:
    # This will catch any other general exceptions
        print(f"Caught an exception: {e}")
        print("An error occurred. Please check your code or environment.")
        
    finally:
        print("Clearing GPU Memory")
        torch.cuda.empty_cache()
