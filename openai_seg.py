import logging
from dataclasses import asdict
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

from openai import OpenAI

from utilities import local_image_to_data_url
from configs.openai_config import openai_api
from prompts import PromptManager
from agent_tools.crop_img import crop_image
from agent_tools.img_outer_border import get_image_with_border
from agent_tools.hough_axis import detect_axis_bounding_box
from agent_tools.plot_extract import extract_and_save_plot_points
from agent_process import *

IMAGE_PATH = "test_figs/given/plot_1.png"
DEBUG = True
CROP_OUTPUT_DIR = "output/cropped"
CROP_OUTPUT_SUFFIX = "{}_cropped"

client = OpenAI(**asdict(openai_api))
data_url = local_image_to_data_url(IMAGE_PATH)
prompt_manager = PromptManager()


def setup_logger(name="my_logger", log_file=None, level=logging.INFO):
    """
    Set up a logger that logs messages at the INFO level and higher.

    Parameters:
        name (str): Name of the logger (default is 'my_logger').
        log_file (str): If provided, log messages will be written to this file.
        level (int): The logging level (default is logging.INFO).

    Returns:
        logging.Logger: The configured logger.
    """
    # Create a logger
    logger = logging.getLogger(name)

    # Set the logging level
    logger.setLevel(level)

    # Create a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler to output logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log file is provided, create a file handler to log to a file
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# print(get_rough_crop(client, data_url, prompt_manager))
logger = setup_logger()

image_with_border = get_image_with_border(
    IMAGE_PATH,
    tick_pixel=50,
    output_dir="test_figs/given",
    output_suffix="with_border",
)
logger.info("Image with border: %s", image_with_border)
border_data_url = local_image_to_data_url(image_with_border)
box_message = get_rough_crop(client, border_data_url, prompt_manager)
box_list = box_message.split("```")[1]
if box_list.startswith("json"):
    box_list = box_list[4:]
box_list = box_list.strip()
try:
    box_list = json.loads(box_list)
    logger.info("Bounding boxes: %s", box_list)
except json.JSONDecodeError:
    logger.error("Error decoding JSON: %s", box_list)
    sys.exit(1)

if DEBUG:
    # plot image with bounding boxes
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    img = Image.open(IMAGE_PATH)
    ax.imshow(np.array(img))
    # Create a Rectangle patch
    for box in box_list:
        top_left = box["top_left"]
        bottom_right = box["bottom_right"]
        rect = patches.Rectangle(
            top_left,
            bottom_right[0] - top_left[0],
            bottom_right[1] - top_left[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
            linestyle="--",
        )
        # draw rectangle with dashed line
        ax.add_patch(rect)
    plt.show()

box_message = box_refine(client, border_data_url, prompt_manager, box_list)
box_list = box_message.split("```")[1]
if box_list.startswith("json"):
    box_list = box_list[4:]
box_list = box_list.strip()
try:
    box_list = json.loads(box_list)
    logger.info("Bounding boxes: %s", box_list)
except json.JSONDecodeError:
    logger.error("Error decoding JSON: %s", box_list)
    sys.exit(1)
logger.info("Bounding boxes after refinement: %s", box_list)

if DEBUG:
    # plot image with bounding boxes
    # Create figure and axes
    fig, ax = plt.subplots()
    # Display the image
    img = Image.open(IMAGE_PATH)
    ax.imshow(np.array(img))
    # Create a Rectangle patch
    for box in box_list:
        top_left = box["top_left"]
        bottom_right = box["bottom_right"]
        rect = patches.Rectangle(
            top_left,
            bottom_right[0] - top_left[0],
            bottom_right[1] - top_left[1],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
            linestyle="--",
        )
        # draw rectangle with dashed line
        ax.add_patch(rect)
    plt.show()
box_info_list = extract_info(client, data_url, box_list, prompt_manager)
# TODO uncapsulate json extraction
box_info_list = box_info_list.split("```")[1]
if box_info_list.startswith("json"):
    box_info_list = box_info_list[4:]
box_info_list = box_info_list.strip()
try:
    box_info_list = json.loads(box_info_list)
except json.JSONDecodeError:
    logger.error("Error decoding JSON: %s", box_info_list)
    sys.exit(1)
logger.info("Extracted information: %s", box_info_list)
crop_image_list = []
for box in box_list:
    cropped_image = crop_image(
        IMAGE_PATH,
        box["top_left"],
        box["bottom_right"],
        CROP_OUTPUT_DIR,
        CROP_OUTPUT_SUFFIX.format(box["chart_id"]),
    )
    crop_image_list.append(cropped_image)
logger.info("Cropped images: %s", crop_image_list)
for crop_image_path, box_info in zip(crop_image_list, box_info_list):
    top_left, bottom_right = detect_axis_bounding_box(
        crop_image_path, "output/borderline", "_bounding_box", DEBUG
    )
    logger.info("Bounding box coordinates: %s, %s", top_left, bottom_right)
    for plot in box_info["plots"]:
        color_name = plot["color"]
        plot_name = plot["name"]
        color = get_color(client, crop_image_path, plot["color"])
        logger.info("Color for plot %s: %s", plot["name"], color)
        extract_and_save_plot_points(
            crop_image_path,
            target_color=(color["R"], color["G"], color["B"]),
            output_dir="output/plot_points",
            output_suffix=f"{plot_name}_plot_points",
            debug=DEBUG,
        )
