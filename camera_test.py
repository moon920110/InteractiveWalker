import pyrealsense2 as rs
import numpy as np
import cv2

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Get RealSense intrinsics
depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

# Depth scaling factor
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

def detect_obstacles(depth_frame, threshold=1.5):
    """Detects obstacles closer than a given threshold (meters)"""
    depth_image = np.asanyarray(depth_frame.get_data())[200:280, 170:270] * depth_scale  # Convert depth to meters
    mask = (depth_image > 0) & (depth_image < threshold)  # Highlight obstacles closer than threshold
    return mask, depth_image
def detect_obstacle_left(depth_frame, threshold=1.5):
    """Detects obstacles closer than a given threshold (meters)"""
    depth_image = np.asanyarray(depth_frame.get_data())[200:280, 170:270] * depth_scale  # Convert depth to meters
    mask = (depth_image > 0) & (depth_image < threshold)  # Highlight obstacles closer than threshold
    return mask, depth_image
def detect_obstacle_mid(depth_frame, threshold=1.5):
    """Detects obstacles closer than a given threshold (meters)"""
    depth_image = np.asanyarray(depth_frame.get_data())[200:280, 270:370] * depth_scale  # Convert depth to meters
    mask = (depth_image > 0) & (depth_image < threshold)  # Highlight obstacles closer than threshold
    return mask, depth_image

def detect_obstacle_right(depth_frame, threshold=1.5):
    """Detects obstacles closer than a given threshold (meters)"""
    depth_image = np.asanyarray(depth_frame.get_data())[200:280, 370:470] * depth_scale  # Convert depth to meters
    mask = (depth_image > 0) & (depth_image < threshold)  # Highlight obstacles closer than threshold
    return mask, depth_image


def compute_distance(depth_image, mask):
    """Compute average distance of obstacles from the camera"""
    obstacle_depth_values = depth_image[mask]
    if len(obstacle_depth_values) > 0:
        average_distance = np.mean(obstacle_depth_values)
        return average_distance
    else:
        return None  # No obstacles

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        # Detect obstacles within 1.5 meters
        obstacle_mask, depth_image = detect_obstacle_left(depth_frame, threshold=1.0)
        # Compute the average distance of obstacles
        left_average_distance = compute_distance(depth_image, obstacle_mask)
        obstacle_mask, depth_image = detect_obstacle_mid(depth_frame, threshold=1.0)
        mid_average_distance = compute_distance(depth_image, obstacle_mask)
        obstacle_mask, depth_image = detect_obstacle_right(depth_frame, threshold=1.0)
        right_average_distance = compute_distance(depth_image, obstacle_mask)
        obstacle_mask, depth_image = detect_obstacles(depth_frame, threshold=1.0)

        # Normalize depth image for visualization
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=50), cv2.COLORMAP_JET)

        # Mark obstacles within 3 meters in red
        depth_colormap[obstacle_mask] = [0, 0, 255]

        # Display depth map with obstacle marking
        cv2.imshow("Depth Map with Obstacles within 3m", depth_colormap)

        # Display the average distance of obstacles within 3 meters
        if left_average_distance is not None:
            print(f"Average distance of left obstacles within 1.5 meters: {left_average_distance:.2f} meters")
        if mid_average_distance is not None:
            print(f"Average distance of mid obstacles within 1.5 meters: {mid_average_distance:.2f} meters")
        if right_average_distance is not None:
            print(f"Average distance of right obstacles within 1.5 meters: {right_average_distance:.2f} meters")
        # else:
        #     print("No obstacles within 1.5 meters detected.")

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")
    pipeline.stop()
    cv2.destroyAllWindows()