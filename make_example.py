from PIL import Image
import h5py
import numpy as np
import imageio

left_image_path = "/root/nrkim/DepthCov/dataset/examples/frame-000338.color.jpg"
right_image_path = "/root/nrkim/DepthCov/dataset/nyu_depth_v2/data/nyu2_test/01420_colors.png"

left_image = Image.open(left_image_path)
right_image = Image.open(right_image_path)

target_size = left_image.size

right_width, right_height = right_image.size

left = (right_width - target_size[0]) // 2
top = (right_height - target_size[1]) // 2
right = (right_width + target_size[0]) // 2
bottom = (right_height + target_size[1]) // 2

right_image_cropped = right_image.crop((left, top, right, bottom))

right_image_resized_jpg_path = "/root/nrkim/DepthCov/dataset/examples/frame-01420.color.jpg"
right_image_cropped.save(right_image_resized_jpg_path, "JPEG")

color_image_path = '/root/nrkim/DepthCov/dataset/nyu_depth_v2/data/nyu2_test/01420_colors.png'
depth_image_path = '/root/nrkim/DepthCov/dataset/nyu_depth_v2/data/nyu2_test/01420_depth.png'
h5_file_path = '/root/nrkim/DepthCov/dataset/nyu_depth_v2/h5_test/01420.h5'
output_pgm_path = '/root/nrkim/DepthCov/dataset/examples/frame-01420.depth.pgm'

color_image = Image.open(color_image_path)

depth_image = Image.open(depth_image_path)

with h5py.File(h5_file_path, 'r') as f:
    depth_data = f['depth'][:]

if depth_data.ndim == 3:
    depth_data = depth_data.squeeze()

depth_data_normalized = (depth_data / np.max(depth_data) * 255).astype(np.uint8)

imageio.imwrite(output_pgm_path, depth_data_normalized, format='pgm')

print(f"PGM file has been created: {output_pgm_path}")
