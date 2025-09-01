import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import datetime
# Current Tests
# 0N = 0.0N
# 2N = 2.0N
# 4N = 4.1N
# 6N = 6.2N
# 8N = 7.7N
# 10N = 9.3N
# --- 1. User Defined Parameters ---
VIDEO_PATH = 'videos/10N.mp4'  # <<< CHANGE THIS TO YOUR VIDEO FILE
DISPLAY_SCALE_FACTOR = 0.75
FRAME_SKIP = 5

# ROI (Region of Interest) cropping
# Set to None to disable cropping, or set to (x, y, width, height)
ROI = (140, 0, 720, 720)  # Example: Crop starting at (140,0) with 720x720 area

# Real-world size of the ROI in millimeters (based on your tactile surface size)
REAL_WORLD_ROI_WIDTH_MM = 20   # mm — physical width of the cropped ROI
REAL_WORLD_ROI_HEIGHT_MM = 20  # mm — physical height of the cropped ROI

# --- CALIBRATION (Quadratic Model) ---
# Uses a second-order polynomial to correct for non-linear sensor response.
# Calibrated Force = A * (raw_force^2) + B * raw_force + C
# Coefficients are derived from a best-fit curve of the user's experimental data.
POLY_COEFF_A = 2.83
POLY_COEFF_B = 6.15
POLY_COEFF_C = -1.25

# Maximum expected height deformation in millimeters (conceptual, used for scaling)
max_height = 4  # mm (used to scale simulated pressure)

# Material stiffness (Young's Modulus) — simulated FEM parameter
YOUNGS_MODULUS = 1e5  # Pa (100 kPa for a soft gel)

# Approximate camera intrinsics (random for now; replace with real values if known)
FX = np.random.uniform(500, 1000)
FY = np.random.uniform(500, 1000)
CX = None
CY = None
K1, K2, P1, P2 = np.random.uniform(-0.1, 0.1, 4)
DIST_COEFFS = np.array([K1, K2, P1, P2])

# Light vectors from photometric stereo calibration (R/G/B light directions)
normal_light_vectors = np.array([
    [0, -4.35, 17.35],
    [0, -4.35, -17.35],
    [17.35, -4.35, 0],
])
inverse_light_vectors = np.linalg.pinv(normal_light_vectors)

# --- 2. Derived Parameters: Computed from ROI and real-world size ---

# Convert mm to meters
REAL_WORLD_ROI_WIDTH_M = REAL_WORLD_ROI_WIDTH_MM / 1000
REAL_WORLD_ROI_HEIGHT_M = REAL_WORLD_ROI_HEIGHT_MM / 1000

# ROI pixel dimensions (used to calculate physical pixel area)
ROI_WIDTH_PX = ROI[2]
ROI_HEIGHT_PX = ROI[3]

# Physical size of a pixel in meters (assuming uniform scale)
PIXEL_WIDTH_M = REAL_WORLD_ROI_WIDTH_M / ROI_WIDTH_PX
PIXEL_HEIGHT_M = REAL_WORLD_ROI_HEIGHT_M / ROI_HEIGHT_PX
PIXEL_AREA_M2 = PIXEL_WIDTH_M * PIXEL_HEIGHT_M  # Area of one pixel in m²

# --- 2. Photometric Stereo & Normal Estimation ---
def estimate_normals_photometric_stereo(image_rgb, light_vectors, inverse_light_vectors):
    h, w, _ = image_rgb.shape
    img_float = image_rgb.astype(np.float32) / 255.0

    intensities = img_float.reshape(-1, 3)
    G_vectors = np.dot(intensities, inverse_light_vectors.T)

    albedo = np.linalg.norm(G_vectors, axis=1)
    albedo_safe = np.where(albedo == 0, 1e-6, albedo)
    normals = G_vectors / albedo_safe[:, np.newaxis]

    normals = normals.reshape(h, w, 3)
    albedo = albedo.reshape(h, w)

    normals = normals / np.linalg.norm(normals, axis=2)[:,:,np.newaxis]
    return normals, albedo

# --- 3. Simulated Inverse FEM Force Estimation ---
def estimate_forces_from_normals(current_normals, reference_normals, gel_stiffness=YOUNGS_MODULUS, max_height=max_height):
    dot_product = np.sum(current_normals * reference_normals, axis=2)
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle_change = np.arccos(dot_product)

    simulated_pressure_map = (angle_change / np.pi) * max_height * gel_stiffness / 1000
    simulated_pressure_map = np.clip(simulated_pressure_map, 0, 200)

    shear_vector_mag = np.linalg.norm(current_normals[:,:,:2] - reference_normals[:,:,:2], axis=2)
    simulated_shear_magnitude_map = shear_vector_mag * gel_stiffness / 500
    simulated_shear_magnitude_map = np.clip(simulated_shear_magnitude_map, 0, 100)

    return simulated_pressure_map, simulated_shear_magnitude_map

# --- 4. Crop to ROI ---
def crop_to_roi(image, roi):
    if roi is None:
        return image
    x, y, w, h = roi
    return image[y:y+h, x:x+w]

# --- Main Script Execution ---
def run_visuotactile_analysis_realtime(video_path, frame_skip, light_vectors, inverse_light_vectors, display_scale_factor, roi=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps
    duration_str = str(datetime.timedelta(seconds=int(duration_sec)))

    # Set up intrinsic matrix dynamically
    global CX, CY, CAMERA_MATRIX
    CX = actual_width / 2
    CY = actual_height / 2
    CAMERA_MATRIX = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])


    # Compute pixel area in meters² based on real-world ROI size
    roi_width_px = roi[2]
    roi_height_px = roi[3]
    roi_width_m = REAL_WORLD_ROI_WIDTH_MM / 1000.0
    roi_height_m = REAL_WORLD_ROI_HEIGHT_MM / 1000.0
    pixel_width_m = roi_width_m / roi_width_px
    pixel_height_m = roi_height_m / roi_height_px
    pixel_area_m2 = pixel_width_m * pixel_height_m

    print(f"Pixel area: {pixel_area_m2:.2e} m²")

    # Read first frame and compute reference normals
    ret, first_frame_original = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return

    first_frame = crop_to_roi(first_frame_original, roi)
    reference_normals, _ = estimate_normals_photometric_stereo(first_frame, light_vectors, inverse_light_vectors)
    print("Reference normals estimated.")

    frame_count = 0
    max_raw_force = 0.0
    max_calibrated_force = 0.0

    while True:
        ret, current_frame_original = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        current_frame = crop_to_roi(current_frame_original, roi)
        current_normals, _ = estimate_normals_photometric_stereo(current_frame, light_vectors, inverse_light_vectors)
        pressure_map, shear_map = estimate_forces_from_normals(current_normals, reference_normals)

        # Normalize and convert pressure map to color
        max_pressure_val = np.max(pressure_map)
        norm_pressure = Normalize(vmin=0, vmax=max_pressure_val * 0.7 if max_pressure_val > 0 else 1)
        pressure_colored = plt.cm.plasma(norm_pressure(pressure_map))[:, :, :3]

        # Compute total normal force in Newtons
        force_map = pressure_map * 1000 * pixel_area_m2
        raw_total_force_n = np.sum(force_map)
        calibrated_total_force_n = (POLY_COEFF_A * (raw_total_force_n**2)) + (POLY_COEFF_B * raw_total_force_n) + POLY_COEFF_C
        calibrated_total_force_n = max(0, calibrated_total_force_n) # Clip at zero to prevent negative force readings

        # Track maximum forces
        if raw_total_force_n > max_raw_force:
            max_raw_force = raw_total_force_n
        if calibrated_total_force_n > max_calibrated_force:
            max_calibrated_force = calibrated_total_force_n

        # Combine views
        vis_image_left = current_frame
        vis_image_right = (pressure_colored * 255).astype(np.uint8)
        combined_vis_image = np.hstack((vis_image_left, vis_image_right))

        display_width = int(combined_vis_image.shape[1] * display_scale_factor)
        display_height = int(combined_vis_image.shape[0] * display_scale_factor)
        display_image = cv2.resize(combined_vis_image, (display_width, display_height))

        # Overlay text
        font_scale = 0.6 * display_scale_factor
        thickness = max(1, int(2 * display_scale_factor))

        # Time and frame display
        current_time_sec = frame_count / fps
        current_time_str = str(datetime.timedelta(seconds=int(current_time_sec)))

        cv2.putText(display_image, "Cropped ROI", (int(10 * display_scale_factor), int(30 * display_scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        cv2.putText(display_image, "Simulated Normal Force (kPa)",
                    (int((vis_image_left.shape[1] + 10) * display_scale_factor), int(30 * display_scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        cv2.putText(display_image,
                    f"Frame: {frame_count}  Time: {current_time_str} / {duration_str}",
                    (int(10 * display_scale_factor), int((combined_vis_image.shape[0] - 10) * display_scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        # Display max pressure and total contact force
        cv2.putText(display_image,
                    f"Total Force: {calibrated_total_force_n:.2f} N",
                    (int((vis_image_left.shape[1] + 10) * display_scale_factor), int((combined_vis_image.shape[0] - 40) * display_scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        cv2.putText(display_image,
                    f"Max Pressure: {max_pressure_val:.2f} kPa",
                    (int((vis_image_left.shape[1] + 10) * display_scale_factor), int((combined_vis_image.shape[0] - 10) * display_scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        cv2.imshow('Visuotactile Force Simulation', display_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDisclaimer: This script uses simplified approximations for FEM.")
    print("Accurate photometric stereo requires careful setup and assumption validation.")
    print("A real implementation requires precise calibration, advanced computer vision, and dedicated FEM solvers.")
    print("\n--- Calibration Info ---")
    print(f"Max Raw Force Detected: {max_raw_force:.4f} N")
    print(f"Max Calibrated Force: {max_calibrated_force:.4f} N")
    print("------------------------\n")

# --- Run the analysis ---
if __name__ == "__main__":
    run_visuotactile_analysis_realtime(
        VIDEO_PATH,
        FRAME_SKIP,
        normal_light_vectors,
        inverse_light_vectors,
        DISPLAY_SCALE_FACTOR,
        ROI
    )
