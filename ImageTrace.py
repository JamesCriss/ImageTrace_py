from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageOps
import numpy as np
from scipy.ndimage import label,convolve
from scipy.spatial import KDTree
import svgwrite


def sort_contour(edges):
    edge_points = np.column_stack(np.where(edges))
    if edge_points.size == 0:
        return np.array([])
    tree = KDTree(edge_points)
    sorted_edge_points = [edge_points[0]]
    visited = set()
    visited.add(tuple(edge_points[0]))

    for _ in range(1, len(edge_points)):
        last_point = sorted_edge_points[-1]
        distances, indices = tree.query(last_point, k=len(edge_points))
        for idx in indices:
            next_point = tuple(edge_points[idx])
            if next_point not in visited:
                sorted_edge_points.append(edge_points[idx])
                visited.add(next_point)
                break

    return np.array(sorted_edge_points)





kernel = np.array([[1, 1, 1],
                   [1, 0, 1], 
                   [1, 1, 1]], dtype=np.uint8)

# Load image
image_original = Image.open('automobile-5128760_1280.jpg')

scale_factor = 1  # Adjust this factor as needed
new_size = (int(image_original.width * scale_factor), int(image_original.height * scale_factor))
image_reduced_res = image_original.resize(new_size)
# Apply median filter to reduce noise
image_filtered = image_reduced_res.filter(ImageFilter.MedianFilter(size=9))

# Reduce the number of colors to 9 using adaptive palette
num_colors = 9
image_reduced = image_filtered.convert("P", palette=Image.ADAPTIVE, colors=num_colors)

# Convert the reduced image to an RGB mode for processing
image_reduced_rgb = image_reduced.convert("RGB")
# Optionally scale the image to have less resolution
image_reduced_rgb.show()
# Convert image to NumPy array
image_array = np.array(image_reduced_rgb)

# Get unique colors in the image
unique_colors = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)

# Dictionary to store labeled islands and their corresponding colors
island_data = np.zeros((image_array.shape[0], image_array.shape[1], 2), dtype=int)

for color_idx, color in enumerate(unique_colors, start=1):
    # Create a binary mask for the current color
    mask = np.all(image_array == color, axis=-1).astype(np.uint8)
    
    # Label connected components (floating islands)
    labeled_array, num_features = label(mask)
    island_size = np.bincount(labeled_array.flatten())
    for i in range(1, num_features + 1):
        if island_size[i] < 300:
            mask[labeled_array == i] = 0

    labeled_array, num_features = label(mask)
    # Store island index and color index in island_data
    island_data[mask == 1, 0] = labeled_array[mask == 1]  # Island index
    island_data[mask == 1, 1] = color_idx  # Color index

    print(f"Color {tuple(color)} has {num_features} floating islands")

# Show images
#-------------------------------------------

width, height = 788*scale_factor,1280*scale_factor
dwg = svgwrite.Drawing("output.svg", size=(width, height))
for color_idx, color in enumerate(unique_colors, start=1):
    for i in range(1, num_features+1):
        
        island_mask = (island_data[:, :, 0] == i) & (island_data[:, :, 1] == color_idx)
        mask_edge_detection = convolve(island_mask.astype(np.uint8), kernel, mode='constant', cval=0)
        edges = (mask_edge_detection < 8) & (island_mask == 1)
        
        edge_points = np.column_stack(np.where(edges))
        sorted_edge_points = sort_contour(edges)
        
        sorted_contour_image = np.zeros_like(image_array, dtype=np.uint8)
        another_array = []
        for idx, point in enumerate(sorted_edge_points):
            if idx % 60 == 0:
                sorted_contour_image[point[0], point[1]] = [255, 255, 255]
                another_array.append((int(point[1]), int(point[0])))
        dwg.add(dwg.polyline(another_array, stroke="black", fill="white", stroke_width=2))
dwg.save()

print("SVG file saved as output.svg")


