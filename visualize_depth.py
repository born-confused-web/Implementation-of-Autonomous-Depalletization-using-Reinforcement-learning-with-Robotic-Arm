import open3d as o3d

def visualize_point_cloud(ply_path):
    print(f"Loading point cloud from {ply_path}...")
    
    # 1. Read the point cloud file (.ply)
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Check if the point cloud loaded successfully
    if not pcd.has_points():
        print(f"Error: Could not load points from {ply_path}. Make sure the file exists.")
        return

    print(f"Successfully loaded {len(pcd.points):,} points.")
    
    # 2. Visualize the point cloud interactively
    print("Launching Open3D visualizer window...")
    print("Tip: Use your mouse left-click + drag to rotate, right-click + drag to pan, and scroll to zoom.")
    
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="OV7670 3D Point Cloud",
        width=1024,
        height=768,
        left=50,
        top=50
    )

if __name__ == "__main__":
    # Point to the file you saved earlier
    ply_filename = "ov7670_point_cloud.ply"
    visualize_point_cloud(ply_filename)