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

# Deadzone parameter (degrees)
DEADZONE = 2.0

def get_actual_angles(cap_front, cap_back, debug=False):
    ret_f, frame_f = cap_front.read()
    ret_b, frame_b = cap_back.read()
    
    if not ret_f or not ret_b:
        return 0.0, 0.0

    # Apply Gaussian smoothing to reduce noise
    frame_f = cv2.GaussianBlur(frame_f, (5, 5), 0)
    frame_b = cv2.GaussianBlur(frame_b, (5, 5), 0)
# White detection using HSV: white has low saturation + high value
    hsv_f = cv2.cvtColor(frame_f, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2HSV)
    # Front camera: track Green LED in the center    
    green_mask = cv2.inRange(hsv_f,
                         (40, 80, 80),
                         (90, 255, 255))
    green_mask = cv2.GaussianBlur(green_mask, (5, 5), 0)
    h_f, s_f, v_f = cv2.split(hsv_f)
    h_b, s_b, v_b = cv2.split(hsv_b)
    green_value = cv2.bitwise_and(v_f, v_f, mask=green_mask)
    _, _, _, max_loc_green = cv2.minMaxLoc(green_value)
    front_green_pt = max_loc_green

    # Back camera: track Red LED (left) and White LED (right)
    b_b, g_b, r_b = cv2.split(frame_b)
    
    pure_red = cv2.subtract(r_b, cv2.max(b_b, g_b))
    _, _, _, max_loc_red = cv2.minMaxLoc(pure_red)
    back_red_pt = max_loc_red

    
   

    # Tune these thresholds for your lighting / camera:
    SATURATION_MAX = 40   # low saturation (near gray/white)
    VALUE_MIN = 200       # very bright

    white_mask = cv2.inRange(hsv_b,
                         (0, 0, VALUE_MIN),
                         (180, SATURATION_MAX, 255))

    # Optionally smooth the mask a bit to reduce noise
    white_mask = cv2.GaussianBlur(white_mask, (5, 5), 0)

    _, _, _, max_loc_white = cv2.minMaxLoc(white_mask)
    back_white_pt = max_loc_white

    # Red detection using HSV: red has high saturation + specific value range
    BRIGHT_RED_V_MIN = 200  # only accept very bright red
    red_mask1 = cv2.inRange(hsv_b, (0, 120, BRIGHT_RED_V_MIN), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv_b, (160, 120, BRIGHT_RED_V_MIN), (180, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_mask = cv2.GaussianBlur(red_mask, (5, 5), 0)
    _, _, _, max_loc_red = cv2.minMaxLoc(red_mask)
    back_red_pt = max_loc_red

    # Debug visualization
    if debug:
        # Front camera visualization
        frame_f_debug = frame_f.copy()
        cv2.circle(frame_f_debug, front_green_pt, 10, (0, 255, 0), 2)  # Green circle
        cv2.putText(frame_f_debug, f"Green: {front_green_pt}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Front Camera - Green LED Detection", frame_f_debug)
        
        # Back camera visualization
        frame_b_debug = frame_b.copy()
        cv2.circle(frame_b_debug, back_red_pt, 10, (0, 0, 255), 2)  # Red circle
        cv2.circle(frame_b_debug, back_white_pt, 10, (255, 255, 255), 2)  # White circle
        cv2.putText(frame_b_debug, f"Red: {back_red_pt}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_b_debug, f"White: {back_white_pt}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Back Camera - Red & White LED Detection", frame_b_debug)
        cv2.waitKey(1)  # 1ms delay for display update

    # Calculate roll: angle between red (left) and white (right) LEDs
    dy = back_white_pt[1] - back_red_pt[1]
    dx = back_white_pt[0] - back_red_pt[0]
    
    # Calculate pixel distance
    pixel_distance = math.sqrt(dx**2 + dy**2)
    if pixel_distance < 2:
        actual_roll = 0.0
    else:
        angle_rad = math.atan2(dy, dx)
        actual_roll = math.degrees(angle_rad)

    # Calculate pitch: angle between front (green) and average back Y position
    back_avg_y = (back_red_pt[1] + back_white_pt[1]) / 2.0
    dy_pitch = front_green_pt[1] - back_avg_y
    if abs(dy_pitch) < 2:
        raw_pitch = 0.0
    else:
        reference_distance = abs(dx) if abs(dx) > 0 else 1
        angle_rad = math.atan2(dy_pitch, reference_distance)
        raw_pitch = math.degrees(angle_rad)
    
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
        raw_cv_pitch, raw_cv_roll = get_actual_angles(cap_front, cap_back, debug=False)
        
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
                gyro_pitch = drone.get_pitch()
                gyro_roll = drone.get_roll()
                
                pitch_error = gyro_pitch - cv_pitch
                roll_error = gyro_roll - cv_roll
                
                # Apply deadzone to ignore small corrections
                if abs(pitch_error) < DEADZONE:
                    pitch_error = 0.0
                if abs(roll_error) < DEADZONE:
                    roll_error = 0.0
                
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