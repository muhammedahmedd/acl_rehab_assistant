# AI-Powered Squat Rep Counter & Knee Flexion Estimator

This project uses **OpenCV** and **MediaPipe** to track a user's leg position from webcam input,
estimate real-time knee flexion angles, and count correct squat repetitions.

### Purpose
To assist with ACL rehabilitation tracking and fitness form correction by giving instant visual feedback on squat depth.

### Features
- Live webcam feed with pose estimation
- Real-time knee flexion angle display
- Visual squat rep counter
- Feedback for shallow or too-deep squats

### How To Install
1. Clone this repository
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe numpy
3. Run the script:
   ```bash
   python_rehab_assistant.py
5. Press q to quit the live feed.

### How It Works
- MediaPipe detects body landmarks
- Flexion angle is calculated at the knee joint
- Squats are counted when the angle drops above 90° and returns below 40°
- Warnings are displayed for improper form
