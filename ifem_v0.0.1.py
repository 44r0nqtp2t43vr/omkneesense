import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- 1. User Defined Parameters ---
VIDEO_PATH = 'videos/sample_dino.mp4'  # <<< CHANGE THIS TO YOUR VIDEO FILE

# Display Scaling Factor: Adjust this to fit your screen
# 1.0 means no scaling (original size), 0.5 means half size, 0.25 means quarter size
DISPLAY_SCALE_FACTOR = 0.5 # Adjust as needed (e.g., 0.75, 0.5, 0.25)

threshold = 0.05
object_distance = -62.4
max_height = 4 # Maximum height/deformation expected, useful for scaling

# Light vectors for photometric stereo (from your calibration)
normal_light_vectors = np.array([
    [0, -4.35, 17.35],
    [0, -4.35, -17.35],
    [17.35, -4.35, 0],
])
inverse_light_vectors = np.linalg.pinv(normal_light_vectors)

# Placeholder camera intrinsics (still random, as not provided)
# CX and CY will be dynamically set based on the *actual* frame size
FX = np.random.uniform(500, 1000)
FY = np.random.uniform(500, 1000)
CX = None 
CY = None 
K1, K2, P1, P2 = np.random.uniform(-0.1, 0.1, 4)
DIST_COEFFS = np.array([K1, K2, P1, P2]) # CAMERA_MATRIX will be set dynamically

# Approximate gel material properties (conceptual for FEM simulation)
YOUNGS_MODULUS = 1e5 # Pa (e.g., 100 kPa for a soft gel)
POISSON_RATIO = 0.49 # Close to 0.5 for incompressible materials like gels

FRAME_SKIP = 5 # Process every Nth frame to speed up

# --- 2. Photometric Stereo & Normal Estimation ---
def estimate_normals_photometric_stereo(image_rgb, light_vectors, inverse_light_vectors):
    """
    Estimates surface normals using a simplified photometric stereo approach.
    Assumes image_rgb contains intensity information from different light sources
    (e.g., R, G, B channels corresponding to distinct light source directions).
    """
    h, w, _ = image_rgb.shape
    img_float = image_rgb.astype(np.float32) / 255.0

    intensities = img_float.reshape(-1, 3) # N_pixels x 3 (R, G, B intensities)

    G_vectors = np.dot(intensities, inverse_light_vectors.T) # (N_pixels, 3)

    albedo = np.linalg.norm(G_vectors, axis=1)

    albedo_safe = np.where(albedo == 0, 1e-6, albedo)
    normals = G_vectors / albedo_safe[:, np.newaxis]

    normals = normals.reshape(h, w, 3)
    albedo = albedo.reshape(h, w)
    
    # Re-normalize just in case (numerical stability)
    normals = normals / np.linalg.norm(normals, axis=2)[:,:,np.newaxis]

    return normals, albedo

# --- 3. Simulated Inverse FEM Force Estimation ---
def estimate_forces_from_normals(current_normals, reference_normals, gel_stiffness=YOUNGS_MODULUS, max_height=max_height):
    """
    Simulates normal force (pressure) and shear force based on normal deviation.
    """
    dot_product = np.sum(current_normals * reference_normals, axis=2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    angle_change = np.arccos(dot_product)

    simulated_pressure_map = (angle_change / np.pi) * max_height * gel_stiffness / 1000 # Convert to kPa
    simulated_pressure_map = np.clip(simulated_pressure_map, 0, 200) # Clip for reasonable display

    shear_vector_mag = np.linalg.norm(current_normals[:,:,:2] - reference_normals[:,:,:2], axis=2)
    simulated_shear_magnitude_map = shear_vector_mag * gel_stiffness / 500 # Adjust divisor for scaling
    simulated_shear_magnitude_map = np.clip(simulated_shear_magnitude_map, 0, 100) # Clip

    return simulated_pressure_map, simulated_shear_magnitude_map

# --- Main Script Execution ---
def run_visuotactile_analysis_realtime(video_path, frame_skip, light_vectors, inverse_light_vectors, display_scale_factor):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get actual dimensions of the video frames
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing full video frames with dimensions: {actual_width}x{actual_height}")

    # Dynamically set CX and CY based on actual frame size
    global CX, CY, CAMERA_MATRIX
    CX = actual_width / 2
    CY = actual_height / 2
    CAMERA_MATRIX = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
    print(f"Camera matrix (dynamic CX, CY): \n{CAMERA_MATRIX}")

    # Read the first frame for reference
    ret, first_frame_original = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        cap.release()
        return
    
    # Process the entire first frame as reference
    reference_normals, _ = estimate_normals_photometric_stereo(first_frame_original, light_vectors, inverse_light_vectors)
    print("Reference normals estimated from the first (full) frame.")

    frame_count = 0
    while True:
        ret, current_frame_original = cap.read()
        if not ret:
            break # End of video

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue # Skip frames to speed up

        # Process the entire current frame
        current_normals, _ = estimate_normals_photometric_stereo(current_frame_original, light_vectors, inverse_light_vectors)

        # Estimate forces (simulated FEM result)
        pressure_map, shear_map = estimate_forces_from_normals(current_normals, reference_normals)

        # --- Visualization ---
        max_pressure_val = np.max(pressure_map)
        norm_pressure = Normalize(vmin=0, vmax=max_pressure_val * 0.7 if max_pressure_val > 0 else 1)
        
        pressure_colored = plt.cm.plasma(norm_pressure(pressure_map))[:,:,:3] # Convert to RGB
        
        # Combine original frame with force maps
        vis_image_left = current_frame_original
        vis_image_right = (pressure_colored * 255).astype(np.uint8)

        combined_vis_image = np.hstack((vis_image_left, vis_image_right))
        
        # --- Apply Display Scaling ---
        display_width = int(combined_vis_image.shape[1] * display_scale_factor)
        display_height = int(combined_vis_image.shape[0] * display_scale_factor)
        
        display_image = cv2.resize(combined_vis_image, (display_width, display_height))

        # Add text overlays for clarity (adjust font scale for smaller display)
        font_scale = 0.6 * display_scale_factor # Scale font size with display
        thickness = max(1, int(2 * display_scale_factor)) # Scale thickness
        
        # Corrected y-coordinate calculations: Ensure the *entire* tuple is type-casted to int at the end
        # The coordinates must be (int_x, int_y)
        cv2.putText(display_image, "Original Full Frame", (int(10 * display_scale_factor), int(30 * display_scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        cv2.putText(display_image, "Simulated Normal Force (kPa)", (int((vis_image_left.shape[1] + 10) * display_scale_factor), int(30 * display_scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
        
        # Corrected line: Ensure the final y-coordinate value is an integer
        # The error was here: `int(combined_vis_image.shape[0] - 10) * display_scale_factor`
        # It should be `int((combined_vis_image.shape[0] - 10) * display_scale_factor)`
        cv2.putText(display_image, f"Frame: {frame_count}", (int(10 * display_scale_factor), int((combined_vis_image.shape[0] - 10) * display_scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Corrected line: Same issue as above
        cv2.putText(display_image, f"Max Pressure: {max_pressure_val:.2f} kPa", (int((vis_image_left.shape[1] + 10) * display_scale_factor), int((combined_vis_image.shape[0] - 10) * display_scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        cv2.imshow('Visuotactile Force Simulation (Real-time)', display_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nDisclaimer: This script uses simplified approximations for FEM.")
    print("Accurate photometric stereo requires careful setup and assumption validation.")
    print("A real implementation requires precise calibration, advanced computer vision, and dedicated FEM solvers.")


# --- Run the analysis ---
if __name__ == "__main__":
    run_visuotactile_analysis_realtime(VIDEO_PATH, FRAME_SKIP, normal_light_vectors, inverse_light_vectors, DISPLAY_SCALE_FACTOR)