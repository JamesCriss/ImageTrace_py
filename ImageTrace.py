from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageOps
import numpy as np
from scipy.ndimage import label


# Load image
image_original = Image.open('automobile-5128760_1280.jpg')

# Apply median filter to reduce noise
image_filtered = image_original.filter(ImageFilter.MedianFilter(size=3))

# Reduce the number of colors to 9 using adaptive palette
num_colors = 2
image_reduced = image_filtered.convert("P", palette=Image.ADAPTIVE, colors=num_colors)

# Convert the reduced image to an RGB mode for processing
image_reduced_rgb = image_reduced.convert("RGB")

# Convert image to NumPy array
image_array = np.array(image_reduced_rgb)

# Get unique colors in the image
unique_colors = np.unique(image_array.reshape(-1, image_array.shape[2]), axis=0)


# Dictionary to store labeled islands and their corresponding colors
island_data = np.zeros((image_array.shape[0], image_array.shape[1], 2), dtype=int)

for color_idx, color in enumerate(unique_colors, start=1):
    # Create a binary mask for the current color
    print("coloridx:", color_idx)
    mask = np.all(image_array == color, axis=-1).astype(np.uint8)
    
    # Label connected components (floating islands)
    labeled_array, num_features = label(mask)
    island_size = np.bincount(labeled_array.flatten())
    for i in range(1, num_features + 1):
        if island_size[i] < 100:
            mask[labeled_array == i] = 0

    labeled_array, num_features = label(mask)
    # Store island index and color index in island_data
    island_data[mask == 1, 0] = labeled_array[mask == 1]  # Island index
    island_data[mask == 1, 1] = color_idx  # Color index

    island_pixels = len(island_data[:, :, 0])
    print("Island pixels:", island_pixels)
    print(f"Color {tuple(color)} has {num_features} floating islands")

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
