import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import depth_pro

def save_depth_map_to_ply(depth_map, rgb_image, output_ply_path, f_px, downsample_factor=1):
    """
    Converts a metric depth map and RGB image into a colored 3D point cloud (.ply file).
    
    Args:
        depth_map (np.ndarray): 2D array of depths in meters, shape (H, W).
        rgb_image (PIL.Image or np.ndarray): Corresponding RGB image.
        output_ply_path (str): File path to save the .ply file.
        f_px (float): Focal length in pixels used during inference.
        downsample_factor (int): Skip pixels to reduce point cloud size (1 = keep all).
    """
    # Ensure RGB image matches depth map dimensions
    if isinstance(rgb_image, Image.Image):
        rgb_image = rgb_image.resize((depth_map.shape[1], depth_map.shape[0]))
    rgb_np = np.array(rgb_image)

    height, width = depth_map.shape
    
    # Create coordinate grid for all pixels
    u_coords, v_coords = np.meshgrid(np.arange(0, width, downsample_factor), 
                                     np.arange(0, height, downsample_factor))
    
    # Extract corresponding depths and RGB colors
    z_vals = depth_map[v_coords, u_coords]
    colors = rgb_np[v_coords, u_coords]

    # Filter out invalid or zero-depth points
    valid_mask = z_vals > 0
    z_vals = z_vals[valid_mask]
    u_coords = u_coords[valid_mask]
    v_coords = v_coords[valid_mask]
    colors = colors[valid_mask]

    # Principal point (optical center assumed at image center)
    c_x = width / 2.0
    c_y = height / 2.0

    # Back-project 2D pixels to 3D points (Camera coordinate system)
    x_vals = (u_coords - c_x) * z_vals / f_px
    y_vals = (v_coords - c_y) * z_vals / f_px
    # In standard camera coordinates, Z points forward, Y points down, X points right
    
    # Stack coordinates and colors into a single array
    points = np.stack((x_vals, y_vals, z_vals), axis=-1)
    
    # Write to PLY file format
    num_points = points.shape[0]
    print(f"Saving {num_points} points to {output_ply_path}...")
    
    with open(output_ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
            
    print("Point cloud saved successfully!")

def get_ov7670_focal_length_px(image_width):
    """
    Calculates the focal length in pixels for the OV7670 sensor.
    Standard OV7670 specs:
      - Lens Focal Length (f_mm): 3.6 mm
      - Sensor Pixel Size: 3.6 um (0.0036 mm)
    
    Formula: f_px = (f_mm / pixel_size_mm) * (current_width / native_width)
    """
    f_mm = 3.6 
    pixel_size_mm = 0.0036
    native_width = 480.0
    
    # Base focal length at native resolution
    native_f_px = f_mm / pixel_size_mm 
    
    # Scale focal length to match the scale of the image width provided to the model
    f_px = native_f_px * (image_width / native_width)
    return f_px

def main():
    # 1. Setup paths and input image
    image_path = "dataset/218_left.jpg"  
    output_path = "ov7670_depth_map.png"
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}. Please provide a valid path.")

    # 2. Load the model and preprocessing transform
    print("Loading DepthPro model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")
    model, transform = depth_pro.create_model_and_transforms()
    model = model.to(device)
    model.eval()

    # 3. Load, rotate, and preprocess the image using PIL
    pil_image = Image.open(image_path).convert("RGB")
    pil_image_rotated = pil_image.rotate(90, expand=True)
    
    transformed_image = transform(pil_image_rotated)
    channels, height, width = transformed_image.shape

    # 4. Calculate actual physical focal length in pixels for OV7670
    custom_f_px = 845.7
    print(f"Bypassing prediction. Using OV7670 physical focal length: {custom_f_px:.2f} px")

    # 5. Run inference with overridden focal length tensor
    batched_image = transformed_image.unsqueeze(0).to(device)
    f_px_tensor = torch.tensor([custom_f_px], device=device)
    
    print("Running depth estimation...")
    with torch.no_grad():
        prediction = model.infer(batched_image, f_px=f_px_tensor)

    # 6. Extract metric depth (values represent distance in meters)
    depth = prediction["depth"].cpu().numpy()
    if depth.ndim == 3:
        depth = depth.squeeze(0)

    # 7. Save and visualize result
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("OV7670 Input (Rotated)")
    plt.imshow(pil_image_rotated)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Metric Depth Map (Meters)")
    plt.imshow(depth, cmap="plasma")
    plt.colorbar(label="Distance (m)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Depth map successfully saved to {output_path}")
    # 8. Export to 3D Point Cloud (.ply)
    output_ply_path = "ov7670_point_cloud.ply"
    save_depth_map_to_ply(
        depth_map=depth, 
        rgb_image=pil_image_rotated, 
        output_ply_path=output_ply_path, 
        f_px=custom_f_px, 
        downsample_factor=1  # Set to 1 for full resolution, or higher (e.g., 2, 4) to make files lighter
    )

if __name__ == "__main__":
    main()