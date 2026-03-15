import cv2
import numpy as np
import math

def find_leds_dual_camera():
    # Open cameras on port 0 and port 1
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    
    if not cap0.isOpened():
        print("Error: Could not open camera on port 0")
        return
    
    if not cap1.isOpened():
        print("Warning: Could not open camera on port 1")
    
    print("Press 'q' to quit")
    
    # Create morphological kernel for better blob detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    while True:
        # Read from both cameras
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0:
            print("Error: Failed to read from camera 0")
            break
        
        # Process camera 0
        result_frame0 = process_frame(frame0, kernel)
        
        # Process camera 1 if available
        if ret1:
            result_frame1 = process_frame(frame1, kernel)
        else:
            result_frame1 = None
        
        # Display both frames side by side
        if result_frame1 is not None:
            combined = np.hstack([result_frame0, result_frame1])
            cv2.imshow('Camera 0 (Left) | Camera 1 (Right)', combined)
        else:
            cv2.imshow('Camera 0', result_frame0)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

def process_frame(frame, kernel):
    """Process a single frame to detect blue and green LEDs"""
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Extract channels (OpenCV uses BGR)
    blue_channel = blurred[:, :, 0].astype(np.float32)
    green_channel = blurred[:, :, 1].astype(np.float32)
    red_channel = blurred[:, :, 2].astype(np.float32)
    
    # Calculate blueness and greenness
    blueness = blue_channel - np.maximum(red_channel, green_channel)
    greenness = green_channel - np.maximum(red_channel, blue_channel)
    
    # Use threshold instead of exact max (top 10% of values)
    blue_threshold = np.percentile(blueness, 90)
    green_threshold = np.percentile(greenness, 90)
    
    # Create masks with threshold
    blue_mask = ((blueness >= blue_threshold) & (blue_channel > 100)).astype(np.uint8) * 255
    green_mask = ((greenness >= green_threshold) & (green_channel > 100)).astype(np.uint8) * 255
    
    # Apply morphological operations to connect nearby pixels
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply dilation to make LEDs more visible
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=1)
    green_mask = cv2.dilate(green_mask, kernel, iterations=1)
    
    # Find contours for each LED
    blue_cx, blue_cy = find_largest_blob(blue_mask)
    green_cx, green_cy = find_largest_blob(green_mask)
    
    # Calculate angle between the two points
    dy = green_cy - blue_cy
    dx = green_cx - blue_cx
    if dx != 0 or dy != 0:
        angle = math.degrees(math.atan2(dy, dx))
    else:
        angle = 0
    
    # Calculate distance
    distance = math.sqrt(dx**2 + dy**2)
    
    # Draw on the frame
    result_frame = frame.copy()
    
    # Draw blue LED center (blue circle)
    if blue_cx > 0 and blue_cy > 0:
        cv2.circle(result_frame, (blue_cx, blue_cy), 8, (255, 0, 0), -1)
        cv2.circle(result_frame, (blue_cx, blue_cy), 15, (255, 0, 0), 2)
    
    # Draw green LED center (green circle)
    if green_cx > 0 and green_cy > 0:
        cv2.circle(result_frame, (green_cx, green_cy), 8, (0, 255, 0), -1)
        cv2.circle(result_frame, (green_cx, green_cy), 15, (0, 255, 0), 2)
    
    # Draw line connecting them
    if blue_cx > 0 and green_cx > 0:
        cv2.line(result_frame, (blue_cx, blue_cy), (green_cx, green_cy), (255, 255, 0), 2)
    
    # Add text with angle and distance
    cv2.putText(result_frame, f"Angle: {angle:.1f}deg", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(result_frame, f"Distance: {distance:.1f}px", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Print information
    print(f"Blue: ({blue_cx}, {blue_cy}) | Green: ({green_cx}, {green_cy}) | Angle: {angle:.1f}° | Distance: {distance:.1f}px", end='\r')
    
    return result_frame

def find_largest_blob(mask):
    """Find the center of the largest blob in a binary mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0, 0
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments to find center
    moments = cv2.moments(largest_contour)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
    else:
        cx, cy = 0, 0
    
    return cx, cy

if __name__ == "__main__":
    find_leds_dual_camera()