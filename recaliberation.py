import cv2
import numpy as np

# 1. CONFIGURATION
CHECKERBOARD = (6, 8)
SQUARE_SIZE = 0.024
calib_file = 'stereo_calibration_params.npz'

# Index configuration for multiple images
START_INDEX = 216
STOP_INDEX = 217  # Inclusive

# Load original calibration parameters
calib_data = np.load(calib_file)
mtx_l = calib_data['mtx_l']
dist_l = calib_data['dist_l']
mtx_r = calib_data['mtx_r']
dist_r = calib_data['dist_r']

# Prepare arrays to store 3D points and 2D points from all valid images
objpoints = []  # 3d point in real world space
imgpoints_l = []  # 2d points in image plane for left camera
imgpoints_r = []  # 2d points in image plane for right camera

# Base 3D object points structure for a single grid
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
img_shape = None

# 2. LOOP AND EXTRACT CORNERS FOR MULTIPLE IMAGES
print(f"Processing images from index {START_INDEX} to {STOP_INDEX}...")

for i in range(START_INDEX, STOP_INDEX + 1):
    # Construct filenames dynamically based on index formatting
    left_path = f'dataset/{i}_left.jpg'
    right_path = f'dataset/{i}_right.jpg'
    
    imgL = cv2.imread(left_path)
    imgR = cv2.imread(right_path)
    
    # Skip if either image is missing
    if imgL is None or imgR is None:
        print(f"Skipping index {i}: Image not found.")
        continue
        
    # Rotate and convert to grayscale
    imgL_rot = cv2.rotate(imgL, cv2.ROTATE_90_COUNTERCLOCKWISE)
    imgR_rot = cv2.rotate(imgR, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    grayL = cv2.cvtColor(imgL_rot, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR_rot, cv2.COLOR_BGR2GRAY)
    
    if img_shape is None:
        img_shape = grayL.shape[::-1]
        
    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, CHECKERBOARD, None)
    retR, cornersR = cv2.findChessboardCorners(grayR, CHECKERBOARD, None)
    
    # If found in both images, refine and append them
    if retL and retR:
        cornersL_sub = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR_sub = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)
        
        objpoints.append(objp)
        imgpoints_l.append(cornersL_sub)
        imgpoints_r.append(cornersR_sub)
        print(f"Successfully processed index {i}")
    else:
        print(f"Skipping index {i}: Checkerboard not detected in both frames.")

# 3. RUN STEREO CALIBRATION ON COLLECTED DATA
if len(objpoints) > 0:
    print(f"\nRecalibrating structural extrinsics using {len(objpoints)} image pairs...")
    
    # cv2.CALIB_FIX_INTRINSIC ensures mtx and dist parameters remain locked
    ret, _, _, _, _, R_new, T_new, E, F = cv2.stereoCalibrate(
        objpoints, 
        imgpoints_l, 
        imgpoints_r, 
        mtx_l, 
        dist_l, 
        mtx_r, 
        dist_r, 
        img_shape, 
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    print(f"Extrinsics updated! New RMS Error: {ret:.4f}")
    
    # 4. RECOMPUTE RECTIFICATION MAPS
    R1_new, R2_new, P1_new, P2_new, Q_new, _, _ = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_shape, R_new, T_new, alpha=0
    )
    
    map1_l_new, map2_l_new = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1_new, P1_new, img_shape, cv2.CV_16SC2)
    map1_r_new, map2_r_new = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2_new, P2_new, img_shape, cv2.CV_16SC2)
    
    # 5. OVERWRITE THE CALIBRATION FILE
    np.savez(
        calib_file,
        mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r,
        R=R_new, T=T_new, R1=R1_new, R2=R2_new, P1=P1_new, P2=P2_new, Q=Q_new,
        map1_l=map1_l_new, map2_l=map2_l_new, map1_r=map1_r_new, map2_r=map2_r_new
    )
    print(f"Successfully saved updated parameters to {calib_file}!")
else:
    print("\nError: No valid checkerboard pairs were found. Calibration aborted.")
