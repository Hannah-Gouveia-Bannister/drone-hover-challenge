# drone-hover-challenge

The code implements a **gyro drift correction system** for a drone using computer vision. Here's the simple explanation:

To run the code, run hover_control.py

**The Problem:**
Drones have gyroscopes that measure rotation, but gyroscopes slowly drift over time (they lose accuracy). The drone needs to correct this drift to hover straight.

**The Solution:**
Use cameras with colored LEDs to measure the drone's *actual* orientation, then compare it to what the gyro thinks, and correct any difference.

**How it works (step by step):**

1. **Track LEDs with cameras:**
   - Front camera watches a green LED to detect pitch (forward/backward tilt)
   - Back camera watches a red LED (left) and white LED (right) to detect roll (left/right tilt)

2. **Calibrate at startup:**
   - Measure the LED positions when the drone is level
   - Save this as the baseline

3. **During flight (every 10 frames):**
   - Average the last 10 camera measurements to reduce noise
   - Compare camera measurement to gyro measurement
   - Calculate the difference (drift error)
   - Send a correction command to the drone

4. **The correction logic:**
   ```
   If gyro says: "Roll = 10 degrees"
   But camera sees: "Roll = 5 degrees"
   Then gyro has drifted by 5 degrees
   So adjust the drone to correct this drift
   ```

## Technical Tricks Used

**Color Detection (current approach):**
- Convert the camera frames to **HSV**.
- Create **per-color HSV masks** to isolate each LED:
  - **Green LED (front camera)**: Hue 40–90, Sat ≥ 80. Use the **V (brightness)** channel inside the mask and pick the **brightest** green pixel.
  - **Red LED (back camera)**: Use two HSV hue ranges (0–10 and 160–180) to handle red wraparound; require a **high V** (bright red) so only the LED is selected.
  - **White LED (back camera)**: Require **low saturation + high brightness** to isolate the white LED and reject colored light.
- Smooth each binary mask using **Gaussian blur** to reduce noise and make position extraction stable.
- Find each LED’s position by locating the maximum value in the relevant masked image using `cv2.minMaxLoc()`.

**Noise Reduction:**
- Use a circular buffer (deque) to average 10 consecutive frames before processing.
- Apply a **deadzone** for roll (ignore values < 2 pixels) to prevent oscillation from camera noise.
- Only send updates every 10 frames to reduce communication overhead.

**Gyro Drift Correction:**
- Read gyro pitch/roll from the drone (in drone units)
- Read camera pitch/roll (converted to drone units)
- Calculate drift error: `error = gyro_reading - camera_reading`
- Apply error as correction: `adjusted_command = target + error`

**Test Mode:**
- Set `TEST_MODE = True` to run without drone connection (prints measurements to console)
- Set `TEST_MODE = False` to connect to real drone and send commands
