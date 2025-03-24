import gradio as gr
import time
import cv2
import numpy as np
from PIL import Image
import openai

from agent_tools.hough_axis import detect_axis_bounding_box
from utilities.logger import setup_logger

logger = setup_logger(name="dynamic_image_processing", log_file="dynamic_image_processing.log", level="DEBUG")
MODEL_NAME="/mnt/longcontext/models/siyuan/llama3/Qwen2.5-VL-72B-Instruct/"
client = openai.OpenAI(api_key="token-abc123", base_url="http://127.0.0.1:8000/v1")

# Example processing function (you can replace this with your own algorithm)
def process_image_dynamically(image: Image.Image):
    logs = []
    intermediate_images = []

    # print image
    intermediate_images.append(image)
    logs.append("Step 1: Loaded image...")
    logger.info("Step 1: Loaded image...")
    
    time.sleep(1)

    bounding_box = detect_axis_bounding_box(image)
    # draw bounding box on image
    if bounding_box is None:
        logger.info("No bounding box detected.")
        logs.append("No bounding box detected.")
        # if no bounding box, stop processing
        return intermediate_images, "\n".join(logs)
    logger.info(f"Bounding box detected: {bounding_box}")
    bounding_box_image = image.copy()
    # convert to numpy array for cv2
    bounding_box_image = np.array(bounding_box_image)
    # convert to BGR for cv2
    bounding_box_image = cv2.cvtColor(bounding_box_image, cv2.COLOR_RGB2BGR)
    # draw bounding box
    cv2.rectangle(bounding_box_image, (bounding_box[0][0], bounding_box[0][1]), (bounding_box[1][0], bounding_box[1][1]), (255, 0, 0), 2)
    intermediate_images.append(bounding_box_image)
    logs.append("Step 2: Detected bounding box...")
    logs.append(f"Bounding box: {bounding_box}")




    return intermediate_images, "\n".join(logs)

# Gradio UI setup
with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ Dynamic Image Processing Demo")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            run_button = gr.Button("Start Processing")
        with gr.Column():
            output_gallery = gr.Gallery(label="Process Images", columns=3, rows=1, height=300)
            output_text = gr.Textbox(label="Log Output", lines=6)

    run_button.click(fn=process_image_dynamically, 
                     inputs=input_image, 
                     outputs=[output_gallery, output_text])

demo.launch()