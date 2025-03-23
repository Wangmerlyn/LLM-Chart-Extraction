from typing import Union
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def plot_image(image:Union[np.ndarray,str,Image.Image])->None:
    """
    Plot an image using matplotlib.

    Parameters:
        image (Union[np.ndarray, str]): The image to plot. Can be a numpy array or a file path.

    Returns:
        None
    """
    if isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError(f"Image must be a file path (str) or a numpy array (ndarray), got {type(image)} instead.")

    plt.imshow(img)
    plt.axis('off')  # Hide the axes
    plt.show()