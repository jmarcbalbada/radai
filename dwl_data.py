from dotenv import load_dotenv
from roboflow import Roboflow
import os

# Load environment variables from the .env file
load_dotenv()

print("Starting Roboflow dataset download...")  # Add a print statement for debugging
api_key = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key)
project = rf.workspace("radai-id0w7").project("kidney-stone-ultrasound-ecckd-5fdji")
version = project.version(1)
dataset = version.download("yolov8")
