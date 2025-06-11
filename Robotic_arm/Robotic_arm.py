import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 虽然不直接使用，但需要激活 3D 投影

# ---------------------------
# initialize MediaPipe Pose
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, smooth_landmarks=True)

# ---------------------------
#Turn on the video stream of the camera
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# ---------------------------
# Matplotlib 3D Real-time display of Settings
# ---------------------------
plt.ion()  # Enter interactive mode
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the key point connections of the human skeleton (MediaPipe has a total of 33 key points. Here are some of the connections.)
BODY_CONNECTIONS = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
]

def draw_3d_skeleton(landmarks_3d):
    """
    Based on the detected 3D key points, the 3D skeleton is drawn using Matplotlib.
    landmarks_3d is a list, and each element is the coordinate of (x, y, z) (the value of MediaPipe, without physical unit conversion).
    """
    ax.clear()  # Clear the previous drawing
    x_vals, y_vals, z_vals = [], [], []
    for pt in landmarks_3d:
        x_vals.append(pt[0])
        y_vals.append(pt[1])
        z_vals.append(pt[2])
    
    # Draw key points
    ax.scatter(x_vals, y_vals, z_vals, c='r', marker='o')
    
    # Draw
    for connection in BODY_CONNECTIONS:
        p1, p2 = connection
        pt1 = landmarks_3d[p1.value]
        pt2 = landmarks_3d[p2.value]
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], 'b-')
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Real-time 3D human posture")
    plt.draw()
    plt.pause(0.001)

# ---------------------------
# Real-time video processing main loop
# ---------------------------
while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("The camera frame cannot be read")
        break

    # Flip the screen for convenient mirror operation (optional)
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB for use by MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # If key points of the human body are detected
    if results.pose_landmarks:
        landmarks_2d = []  # Store 2D coordinates for video display
        landmarks_3d = []  # Store 3D coordinates for 3D drawing
        h, w, _ = frame.shape

        # Traverse all 33 key points
        for lm in results.pose_landmarks.landmark:
            # 2D coordinates (pixels)
            x2d, y2d = int(lm.x * w), int(lm.y * h)
            landmarks_2d.append((x2d, y2d))
            # 3D coordinates (MediaPipe provides relative values, and the z value is usually small)
            landmarks_3d.append((lm.x, lm.y, lm.z))
        
        # Draw 2D key points and skeletons onto video frames

        for pt in landmarks_2d:
            cv2.circle(frame, pt, 5, (0, 255, 0), -1)
        for connection in BODY_CONNECTIONS:
            pt1 = landmarks_2d[connection[0].value]
            pt2 = landmarks_2d[connection[1].value]
            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
        
        # Update the 3D skeleton visualization
        draw_3d_skeleton(landmarks_3d)
    
    # Display the camera video with 2D skeletons
    cv2.imshow("实时 2D 姿态检测", frame)
    
    # export FPS
    fps = 1.0 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")
    
    # Press the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------
# release resource
# ---------------------------
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
