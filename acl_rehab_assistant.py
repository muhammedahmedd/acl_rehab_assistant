"""
Real-time Knee Flexion Angle Estimation and Squat Rep Counter using OpenCV and MediaPipe.

This script captures live webcam footage, estimates knee joint flexion angles using pose landmarks
from MediaPipe, and counts correct squat repetitions based on angle thresholds. Real-time feedback
is displayed to help adjust squat form for effective and safe ACL rehabilitation.

Author: Mohamed Ahmed
Date: 06/28/2025
"""

import cv2 as cv
import mediapipe as mp
import numpy as np

# ------------------------------
# Utility Functions
# ------------------------------

def calculate_flexion(a, b, c):
    """
    Calculate the knee flexion angle at joint 'b' using three 2D points.

    Flexion is computed as:
        flexion = 180° - angle formed at point b

    Args:
        a (array-like): Coordinates of the hip.
        b (array-like): Coordinates of the knee (joint where angle is calculated).
        c (array-like): Coordinates of the ankle.

    Returns:
        float: Knee flexion angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    flexion = 180 - angle
    return flexion

# ------------------------------
# Initialize MediaPipe Pose
# ------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv.VideoCapture(0)

# ------------------------------
# State Variables
# ------------------------------
correct_reps = 0           # Number of valid squat repetitions
stage = None               # Squat movement stage ("up" or "down")
ready_for_feedback = False # Flag to control feedback rendering

# ------------------------------
# Main Processing Loop
# ------------------------------
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)  # Mirror the image for user-friendly interaction
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec((0, 255, 0), 4, 2),
                                      mp_drawing.DrawingSpec((0, 255, 100), 2, 2))

            # ------------------------------
            # Extract Landmarks for Both Legs
            # ------------------------------
            left_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # ------------------------------
            # Calculate Knee Flexion Angles
            # ------------------------------
            left_angle = calculate_flexion(left_hip, left_knee, left_ankle)
            right_angle = calculate_flexion(right_hip, right_knee, right_ankle)

            feedback = []

            # ------------------------------
            # Squat Detection Logic
            # ------------------------------
            if left_angle > 90:
                stage = "down"
            
            elif left_angle < 40 and stage == "down":
                stage = "up"
                correct_reps += 1
                

            # ------------------------------
            # Real-Time Feedback (Only Hip Depth)
            # ------------------------------
            if 40 < left_angle < 80:
                feedback.append("Lower your hips more to squat")
            elif left_angle > 135:
                feedback.append("Don't go too deep")
            

            frame_h, frame_w, _ = image.shape

            # ------------------------------
            # Display Knee Flexion (Top-Left)
            # ------------------------------
            cv.rectangle(image, (20, 20), (360, 110), (255, 0, 0), -1)
            cv.putText(image, f"Right Knee Flexion: {round(right_angle, 1)} deg", (30, 55),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(image, f"Left Knee Flexion:  {round(left_angle, 1)} deg", (30, 95),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

            # ------------------------------
            # Display Squat Rep Count (Top-Right)
            # ------------------------------
            correct_text = f"Correct Squat Reps: {correct_reps}"
            text_size = cv.getTextSize(correct_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            box_x = frame_w - text_size[0] - 40
            cv.rectangle(image, (box_x - 10, 20), (frame_w - 20, 70), (0, 200, 0), -1)
            cv.putText(image, correct_text, (box_x, 55),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)

            # ------------------------------
            # Display Feedback (Bottom-Left)
            # ------------------------------
            y_offset = frame_h - 100
            for i, msg in enumerate(feedback):
                cv.rectangle(image, (30, y_offset + i * 40 - 30), (500, y_offset + i * 40), (50, 50, 50), -1)
                cv.putText(image, msg, (40, y_offset + i * 40 - 5),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)

        cv.imshow('MediaPipe Feed', image)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

# ------------------------------
# Clean up
# ------------------------------
cap.release()
cv.destroyAllWindows()








