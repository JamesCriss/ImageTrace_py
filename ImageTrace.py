from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageOps
import numpy as np
from scipy.ndimage import label

# Load image
image_original = Image.open('automobile-5128760_1280.jpg')

# Apply median filter to reduce noise
image_filtered = image_original.filter(ImageFilter.MedianFilter(size=9))

# Reduce the number of colors to 9 using adaptive palette
num_colors = 9
image_reduced = image_filtered.convert("P", palette=Image.ADAPTIVE, colors=num_colors)

# Convert the reduced image to an RGB mode for processing
image_reduced_rgb = image_reduced.convert("RGB")

# Convert image to NumPy array
image_array = np.array(image_reduced_rgb)

# Get unique colors in the image
unique_colors = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)

# Dictionary to store labeled islands and their corresponding colors
island_data = np.zeros((image_array.shape[0], image_array.shape[1], 2), dtype=int)

# Threshold for minimum island size
min_pixels = 50

for color_idx, color in enumerate(unique_colors, start=1):
    # Create a binary mask for the current color
    mask = np.all(image_array == color, axis=-1).astype(np.uint8)
    
    # Label connected components (floating islands)
    labeled_array, num_features = label(mask)
    
    # Remove small islands
    for island_id in range(1, num_features + 1):
        if np.sum(labeled_array == island_id) < min_pixels:
            labeled_array[labeled_array == island_id] = 0
    
    # Store island index and color index in island_data
    mask = labeled_array > 0  # Update mask after filtering
    island_data[mask, 0] = labeled_array[mask]  # Island index
    island_data[mask, 1] = color_idx  # Color index
    
    print(f"Color {tuple(color)} has {np.max(labeled_array)} floating islands after filtering")

# Show images
image_reduced.show()
image_filtered.show()

# Display the third island for the first color
first_color = unique_colors[0]
third_island_mask = (island_data[:, :, 0] == 3) & (island_data[:, :, 1] == 1)
third_island_image = np.zeros_like(image_array, dtype=np.uint8)
third_island_image[third_island_mask] = first_color
third_island_pil = Image.fromarray(third_island_image)
third_island_pil.show()
