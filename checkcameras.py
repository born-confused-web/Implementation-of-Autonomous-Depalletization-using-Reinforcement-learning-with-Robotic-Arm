import cv2
import numpy as np

# 1. CONFIGURATION (Matching your original setup)
CHECKERBOARD = (6, 8)  
# Use the correct file name you saved it as
calib_file = 'stereo_calibration_params.npz'

# Load your saved stereo calibration data
calib_data = np.load(calib_file)

# Extract your exact keys
mtx_l = calib_data['mtx_l']   
dist_l = calib_data['dist_l'] 
mtx_r = calib_data['mtx_r']   
dist_r = calib_data['dist_r'] 
R = calib_data['R']         
T = calib_data['T']         

# 2. READ & ROTATE THE NEW IMAGES (Crucial step!)
# Replace these with the actual filenames of your new checkerboard shot
imgL = cv2.imread('dataset/216_left.jpg')
imgR = cv2.imread('dataset/216_right.jpg')

# Match the rotation logic from your calibration setup
imgL_rot = cv2.rotate(imgL, cv2.ROTATE_90_COUNTERCLOCKWISE)
imgR_rot = cv2.rotate(imgR, cv2.ROTATE_90_COUNTERCLOCKWISE)

grayL = cv2.cvtColor(imgL_rot, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR_rot, cv2.COLOR_BGR2GRAY)
h, w = grayL.shape[:2]

# 3. COMPUTE RECTIFICATION MAPS
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, (w, h), R, T, alpha=0)

mapLx, mapLy = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, (w, h), cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, (w, h), cv2.CV_32FC1)

# Rectify the new rotated images
rectL = cv2.remap(imgL_rot, mapLx, mapLy, cv2.INTER_LINEAR)
rectR = cv2.remap(imgR_rot, mapRx, mapRy, cv2.INTER_LINEAR)

# 4. FIND CHECKERBOARD CORNERS
retL, cornersL = cv2.findChessboardCorners(rectL, CHECKERBOARD, None)
retR, cornersR = cv2.findChessboardCorners(rectR, CHECKERBOARD, None)

if retL and retR:
    # Calculate vertical (Y) error between corresponding points
       # Force arrays into standard 2D shape (N, 2) where column 1 is Y-coordinates
    ptsL = cornersL.reshape(-1, 2)
    ptsR = cornersR.reshape(-1, 2)

    # Calculate vertical (Y) error between corresponding points
    y_errors = np.abs(ptsL[:, 1] - ptsR[:, 1])
    mean_y_error = np.mean(y_errors)
    max_y_error = np.max(y_errors)

    print(f"--- Calibration Check Results ---")
    print(f"Mean Vertical Error: {mean_y_error:.3f} pixels")
    print(f"Max Vertical Error:  {max_y_error:.3f} pixels")
    
    if mean_y_error < 1.0:
        print("✅ The calibration is still valid! The alignment is good.")
    else:
        print("❌ The cameras have moved! You need to recalibrate extrinsics.")
        
    # 5. DRAW HORIZONTAL VISUALIZATION LINES
    combined = np.hstack((rectL, rectR))
    # Draw horizontal green lines across the stack every 20 pixels
    for i in range(0, combined.shape[0], 20):
        cv2.line(combined, (0, i), (combined.shape[1], i), (0, 255, 0), 1)
        
    cv2.imshow("Rectified Stereo Check (Lines must cross identical corners)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not find the checkerboard pattern in one or both rectified images.")
    print(f"Left detected: {retL}, Right detected: {retR}")
