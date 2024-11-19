import streamlit as st
import requests
import json
from PIL import Image, ImageDraw, ImageFont

# Title for the app
st.title("RadAI: Kidney Stone Detection")

st.markdown("""
    **RadAI** is an AI-powered binary classification tool designed to assist in detecting kidney stones in ultrasound images. 
    The model classifies images into two categories: **Kidney Stone Detected** or **Normal Kidney**. 
    Simply upload or capture an image, and RadAI will process it and provide real-time results based on the analysis.
""")

# Input option radio
st.subheader("Input Options")
input_option = st.radio("Choose input type", ("Upload Image", "Use Camera"), key="input_radio")

# Dropdown to select the model
model_option = st.selectbox("Select Model", ("YOLOv11x", "YOLOv8x"), key="model_dropdown")

# Initialize variables
img_file_buffer = None
uploaded_image = None

# Use the camera
if input_option == "Use Camera":
    img_file_buffer = st.camera_input("Take a picture")

# Upload image
elif input_option == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Image handling
image_path = None
if img_file_buffer or uploaded_image:
    # Display the captured or uploaded image
    image = img_file_buffer if img_file_buffer else uploaded_image
    st.image(image, caption="Selected Image")

    # Save the image to a file
    image_path = "selected_image.jpg"
    with open(image_path, "wb") as file:
        file.write(image.getbuffer())
        st.success(f"Image saved as '{image_path}'")

# Inference and Overlay
if image_path and st.button("Run Prediction"):
    # URL and API setup based on selected model
    url = "https://predict.ultralytics.com"
    headers = {"x-api-key": "3b5056ac3a9ea918ac838037d777446ba97e9ad3fc"}
    data = {
        "model": "https://hub.ultralytics.com/models/eLi9nXp1Q5RnK0HlHBhn" if model_option == "YOLOv11x" else "https://hub.ultralytics.com/models/SMt917G5PhT5W142f1Iq",
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.45,
    }

    try:
        # Send the image for inference
        with open(image_path, "rb") as f:
            response = requests.post(url, headers=headers, data=data, files={"file": f})
        response.raise_for_status()

        # Parse results
        results = response.json()

        # Load the image
        original_image = Image.open(image_path)
        draw = ImageDraw.Draw(original_image)

        # Optional: Load a font for text (fallback if not found)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        # Flag to check if Kidney Stone or Normal Kidney is detected
        kidney_stone_detected = False
        normal_kidney_detected = False
        total_confidence = 0
        kidney_stone_count = 0
        normal_kidney_count = 0

        # Iterate through the results and calculate confidence
        for detection in results["images"][0]["results"]:
            box = detection["box"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = f'{detection["name"]} ({detection["confidence"]:.2f})'

            # Check if the detected object is a "kidney-stone"
            if "kidney-stone" in detection["name"].lower():
                kidney_stone_detected = True
                total_confidence += detection["confidence"]
                kidney_stone_count += 1

            # Check if the detected object is a "normal kidney"
            elif "normal kidney" in detection["name"].lower():
                normal_kidney_detected = True
                total_confidence += detection["confidence"]
                normal_kidney_count += 1

            # Draw rectangle and label (as in your original code)
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            
            # Adjust the y position for label text to avoid overlap with the bounding box
            label_y = y1 - 15 if y1 - 15 > 0 else y2 + 5
            label_y = label_y if label_y > 5 else y2 + 10  # Additional check to avoid out-of-bounds
            
            # Draw label
            draw.text((x1, label_y), label, fill="red", font=font)

        # Add the result label
        if kidney_stone_detected:
            average_confidence = total_confidence / kidney_stone_count
            result_label = f"Result: Kidney Stone Detected \n Average Confidence: {average_confidence:.2f}"
        elif normal_kidney_detected:
            average_confidence = total_confidence / normal_kidney_count
            result_label = f"Result: Normal Kidney \n Average Confidence: {average_confidence:.2f}"
        else:
            result_label = "Result: No Kidney Detected"

        st.subheader(result_label)


        # Optionally print out the full JSON for debugging
        st.json(results)

        # Convert the image to RGB if it has an alpha channel
        if original_image.mode == "RGBA":
            original_image = original_image.convert("RGB")

        # Save and display the image with annotations
        annotated_image_path = "annotated_image.jpg"
        original_image.save(annotated_image_path)
        st.image(original_image, caption="Annotated Image with Predictions")

        # Optionally download the annotated image
        with open(annotated_image_path, "rb") as f:
            st.download_button(
                label="Download Annotated Image",
                data=f,
                file_name="annotated_image.jpg",
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(f"Error during prediction: {e}")
