"""
This file contains the prompt manager
"""

from dataclasses import dataclass


@dataclass
class PromptManager:
    """
    This class is the prompt manager
    """

    crop_prompt: str = """I will provide you with an image containing multiple charts. Around the charts, there are some rough coordinate annotations or visual indicators that can help identify the approximate boundaries of each chart. Your task is to identify and return the approximate bounding box coordinates (top-left and bottom-right corners) for each individual chart within the image. For each sub-chart in the figure, tell me the height of the upper bound, the height of the lower bound first. Remember, these heights must only align with the coordinate indicators I provided around the image as well as the grid in the figure. Then output the bounding boxes. You are allowed to output boxes that are too big, but you can not output boxes that are too small that it didn't cover the full sub-chart. The format of the output should be a list of dictionaries, where each dictionary represents a chart and contains its coordinates in the following format:
```json
[
  {"chart_id": 1, "top_left": [x1, y1], "bottom_right": [x2, y2]},
  {"chart_id": 2, "top_left": [x1, y1], "bottom_right": [x2, y2]},
  ...
]```
 The bounding boxes can be slightly larger to ensure the entire chart is included.
 """
    box_refine_prompt: str = """I will provide you with an image containing multiple charts. Around the charts, there are some rough coordinate annotations or visual indicators that can help identify the approximate boundaries of each chart.I have a rough bound box estimation: {}. Help me change my bound box estimation based on the following rules: If one of the chart is the top most one, change the upper bound of its bound box to the top of the figure. If one of the chart is the bottom most one, change the lower bound of its bound box to the bottom of the figure.
Additionally, if two charts are vertically or horizontally adjacent (e.g., one chart's bottom boundary is near another chart's top boundary), adjust the boundary coordinate in question to be the average of the two overlapping boundaries. For example, if Chart 1's bottom boundary is at y = 100 and Chart 2's top boundary is at y = 102, adjust both to y = 101..
                    The format of the output should be a list of dictionaries, where each dictionary represents a chart and contains its coordinates in the following format:
```json
[
  {{"chart_id": 1, "top_left": [x1, y1], "bottom_right": [x2, y2]}},
  {{"chart_id": 2, "top_left": [x1, y1], "bottom_right": [x2, y2]}},
  ...
]```
Each coordinate should represent pixel values relative to the original image. The bounding boxes can be slightly larger to ensure the entire chart is included.
"""
    extract_info_prompt: str = """
    I will provide you with a JSON containing refined bounding box coordinates for multiple sub-charts within an image. Your task is to analyze each sub-chart and extract the following details for each one:

Y-Axis Information:

Label: The text label of the y-axis.
Max: The maximum value on the y-axis.
Min: The minimum value on the y-axis.
Ticks: A list of all tick values on the y-axis (if present).
X-Axis Information:

Label: The text label of the x-axis.
Max: The maximum value on the x-axis.
Min: The minimum value on the x-axis.
Ticks: A list of all tick values on the x-axis (if present).
Sub-Chart Title: The title of the sub-chart (if available).

Plot Information (for sub-charts containing multiple plots):

A list of all plot names and their corresponding colors as specified in the legend. Each entry should include:
Plot name: The name of the plot (as listed in the legend).
Color: The color associated with the plot in the legend.
Y Axis Information: The y-axis information for the sub-chart.
X Axis Information: The x-axis information for the sub-chart.

The input format for the bounding boxes is as follows:
{}

For each sub-chart, extract the necessary information by analyzing the content within the provided bounding box. If any required information is missing or unclear, make a note of it and provide your best estimate or explanation. Please be precise with colors and text labels. For example please don't confuse cyan, green and teal.

The output format should be a list of dictionaries, where each dictionary represents a sub-chart and contains the extracted details:

```json
[
  {{
    "chart_id": 1,
    "top_left": [x1, y1],
    "bottom_right": [x2, y2],
    "y_axis_label": "Label text",
    "y_axis_max": 100,
    "y_axis_min": 0,
    "y_axis_ticks": [0, 20, 40, 60, 80, 100],
    "x_axis_label": "Label text",
    "x_axis_max": 10,
    "x_axis_min": 0,
    "x_axis_ticks": [0, 2, 4, 6, 8, 10],
    "title": "Sub-chart title",
    "plots": [
      {{"name": "Plot 1", "color": "red", "y_axis_max": 3, "y_axis_min": 0, y_axis_label: "axis_1_label"}},
      {{"name": "Plot 2", "color": "blue", "y_axis_max": 100, "y_axis_min": 0, y_axis_label: "axis_2_label"}},
    ]
  }},
  {{
    "chart_id": 2,
    ...
  }}
]
```
If a certain axis or title is missing, just output null.
"""
