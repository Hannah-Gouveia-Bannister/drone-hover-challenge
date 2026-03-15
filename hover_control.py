import time
import cv2
import math
from collections import deque

TEST_MODE = False # Set to False to connect and send commands to the drone
FRONT_PORT = 0
BACK_PORT = 1

# Only import drone module if not in test mode
if not TEST_MODE:
    import dronerc_original as drone

# Frame buffering parameters
FRAME_BUFFER_SIZE = 10
pitch_buffer = deque(maxlen=FRAME_BUFFER_SIZE)
roll_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

def get_actual_angles(cap_front, cap_back):
    ret_f, frame_f = cap_front.read()
    ret_b, frame_b = cap_back.read()
    
    if not ret_f or not ret_b:
        return 0.0, 0.0

    # Front camera: track Green LED in the center
    b_f, g_f, r_f = cv2.split(frame_f)
    pure_green = cv2.subtract(g_f, cv2.max(b_f, r_f))
    _, _, _, max_loc_green = cv2.minMaxLoc(pure_green)
    front_green_pt = max_loc_green

    # Back camera: track Red LED (left) and White LED (right)
    b_b, g_b, r_b = cv2.split(frame_b)
    
    pure_red = cv2.subtract(r_b, cv2.max(b_b, g_b))
    _, _, _, max_loc_red = cv2.minMaxLoc(pure_red)
    back_red_pt = max_loc_red

    white_intensity = cv2.subtract(cv2.add(b_b, g_b, dtype=cv2.CV_32F), r_b, dtype=cv2.CV_32F)
    _, _, _, max_loc_white = cv2.minMaxLoc(white_intensity)
    back_white_pt = max_loc_white

    # Calculate actual roll: vertical offset between red (left) and white (right) LEDs
    dy = back_white_pt[1] - back_red_pt[1]
    
    # Convert pixel difference to degrees
    pixel_to_degree_roll = 0.1
    
    # Apply deadzone to reduce noise for small differences
    if abs(dy) < 2:
        actual_roll = 0.0
    else:
        actual_roll = -dy * pixel_to_degree_roll
    
    # Convert from degrees to drone units (multiply by 16)
    actual_roll *= 16

    # Pitch: Difference in Y between the front (green) and the average back Y
    pixel_to_degree_scale = 0.1
    back_avg_y = (back_red_pt[1] + back_white_pt[1]) / 2.0
    raw_pitch = (front_green_pt[1] - back_avg_y) * pixel_to_degree_scale
    
    # Convert from degrees to drone units (multiply by 16)
    raw_pitch *= 16
    
    return raw_pitch, actual_roll

def main():
    # Initialize USB cameras
    cap_front = cv2.VideoCapture(0)
    cap_back = cv2.VideoCapture(1)

    # Base thrust for hovering (needs tuning)
    base_thrust = 200 
    
    if not TEST_MODE:
        # Initialize drone
        drone.set_mode(2)
        
        # Enable LEDs for tracking
        drone.red_LED(1)
        drone.green_LED(1)
        drone.blue_LED(1)
        
        drone.manual_thrusts(base_thrust, base_thrust, base_thrust, base_thrust)
    
    print("Waiting 2.5 seconds for calibration...")
    time.sleep(2.5)
    
    # Flush camera buffers to get the latest frame for calibration
    for _ in range(5):
        cap_front.read()
        cap_back.read()
        
    baseline_pitch, baseline_roll = get_actual_angles(cap_front, cap_back)
    print(f"Calibration complete. Baseline pitch offset: {baseline_pitch:.2f}, Baseline roll offset: {baseline_roll:.2f}")

    target_pitch = 0.0
    target_roll = 0.0

    print("Starting 60-second hover...")
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 60.0:
        raw_cv_pitch, raw_cv_roll = get_actual_angles(cap_front, cap_back)
        
        # Add raw values to buffers
        pitch_buffer.append(raw_cv_pitch)
        roll_buffer.append(raw_cv_roll)
        frame_count += 1
        
        # Only process and send update every 10 frames
        if frame_count % FRAME_BUFFER_SIZE == 0:
            # Calculate average of buffered frames
            avg_pitch = sum(pitch_buffer) / len(pitch_buffer)
            avg_roll = sum(roll_buffer) / len(roll_buffer)
            
            # Calculate deviation from calibrated baseline
            cv_pitch = avg_pitch - baseline_pitch
            cv_roll = avg_roll - baseline_roll
            
            if TEST_MODE:
                print(f"Frame {frame_count}: Averaged Pitch: {cv_pitch:.2f}, Roll: {cv_roll:.2f} (from {len(pitch_buffer)} frames)")
            else:
                gyro_pitch = drone.get_pitch()  # Already in drone units
                gyro_roll = drone.get_roll()    # Already in drone units
                
                # Calculate drift error
                pitch_error = gyro_pitch - cv_pitch
                roll_error = gyro_roll - cv_roll
                
                # Apply the error to our desired target (0 in drone units)
                adjusted_pitch = target_pitch + pitch_error
                adjusted_roll = target_roll + roll_error
                
                drone.set_pitch(adjusted_pitch)
                drone.set_roll(adjusted_roll)
        
        # Prevent constant high-bandwidth communication
        time.sleep(0.1)

    print("Hover complete. Stopping.")
    if not TEST_MODE:
        drone.emergency_stop()
    
    cap_front.release()
    cap_back.release()

if __name__ == "__main__":
    main()