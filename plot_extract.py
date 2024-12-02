import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from langchain import tools


@tools
def extract_and_save_plot_points(
    image_path,
    target_color,
    tolerance=10,
    link_distance=5,
    legend_filter_length=50,
):
    """
    Extracts and links plot points in a chart image based on the specified color, removes legend samples,
    interpolates between segments, and saves the points as a sorted NumPy array.

    Parameters:
        image_path (str): Path to the input image.
        target_color (tuple): The RGB color of the plot to extract (e.g., (0, 255, 0) for green).
        tolerance (int): Tolerance for color matching (default is 10).
        link_distance (int): Maximum distance between points to link fragments (default is 5).
        legend_filter_length (int): Maximum segment length to be considered a legend sample (default is 50 pixels).

    Returns:
        np.ndarray: A 2D NumPy array where each row represents a point (x, y), sorted by x.
    """
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the lower and upper bounds for color matching
    lower_bound = np.array([max(0, c - tolerance) for c in target_color])
    upper_bound = np.array([min(255, c + tolerance) for c in target_color])

    # Create a mask for the target color
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

    # Find the coordinates of the matching pixels
    y_coords, x_coords = np.where(mask > 0)
    plot_points = sorted(
        list(zip(x_coords, y_coords)), key=lambda p: p[0]
    )  # Sort by x-coordinate

    if not plot_points:
        print("No points detected for the specified color.")
        return np.array([])

    # Link nearby points
    linked_points = []
    current_segment = [plot_points[0]]

    for i in range(1, len(plot_points)):
        if (
            cdist([current_segment[-1]], [plot_points[i]])[0][0]
            <= link_distance
        ):
            current_segment.append(plot_points[i])
        else:
            # Save the current segment and start a new one
            linked_points.append(current_segment)
            current_segment = [plot_points[i]]
    if current_segment:
        linked_points.append(current_segment)

    # Filter out potential legend samples
    filtered_segments = []
    for segment in linked_points:
        segment = np.array(segment)
        if len(segment) > 1:
            # Check if the segment is straight (low variance in y-values)
            y_variance = np.var(segment[:, 1])
            x_span = (
                segment[-1, 0] - segment[0, 0]
            )  # Horizontal length of the segment
            if y_variance > 1 or x_span > legend_filter_length:
                filtered_segments.append(segment)

    # Interpolate to connect separate segments
    final_points = []
    for i in range(len(filtered_segments) - 1):
        segment1 = filtered_segments[i]
        segment2 = filtered_segments[i + 1]

        # Add the points from the first segment
        final_points.extend([tuple(point) for point in segment1])

        # Interpolate between the last point of segment1 and the first point of segment2
        x_interp = np.linspace(
            segment1[-1, 0], segment2[0, 0], num=10, endpoint=True
        )
        y_interp = np.linspace(
            segment1[-1, 1], segment2[0, 1], num=10, endpoint=True
        )
        final_points.extend(
            list(zip(x_interp.astype(int), y_interp.astype(int)))
        )

    # Add the last segment's points
    if filtered_segments:
        final_points.extend([tuple(point) for point in filtered_segments[-1]])

    # Convert to NumPy array and sort by x-coordinate
    final_array = np.array(final_points)
    final_array = final_array[
        np.argsort(final_array[:, 0])
    ]  # Sort by x-coordinate

    # Visualize the original image and extracted points in subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Original image with highlighted points
    output_image = image_rgb.copy()
    for x, y in final_points:
        cv2.circle(
            output_image, (x, y), 1, (255, 0, 0), -1
        )  # Highlight points in red
    axes[0].imshow(output_image)
    axes[0].set_title("Original Image with Interpolated Points")
    axes[0].axis("off")

    # Plot only the extracted points
    axes[1].scatter(final_array[:, 0], final_array[:, 1], s=1, color="red")
    axes[1].set_title("Extracted and Sorted Plot Points")
    axes[1].invert_yaxis()  # Flip y-axis to match image orientation
    axes[1].set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.show()

    return final_array


def main():
    # Example usage
    image_path = "test_figs/given/plot_0_1.png"
    target_color = (0, 255, 255)  # Example: cyan
    tolerance = 20  # Adjust tolerance as needed
    link_distance = 5  # Maximum distance to link fragments
    legend_filter_length = 50  # Maximum length to identify legend lines
    linked_points_array = extract_and_save_plot_points(
        image_path, target_color, tolerance, link_distance, legend_filter_length
    )
    print(linked_points_array[:10])  # Display the first 10 rows
    print(linked_points_array.shape)  # Display the array shape
    # save the points to a file
    np.savetxt(
        "test_figs/given/plot_0_1_points.csv",
        linked_points_array,
        delimiter=",",
    )


if __name__ == "__main__":
    main()
