import cv2
import mediapipe as mp
import numpy as np
import time
import winsound
import threading

# Import Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# from mediapipe.framework.formats import landmark_pb2

# Configuration
MODEL_PATH = 'face_landmarker.task'

# Global variables for state
sleep = 0
drowsy = 0
active = 0
status = "Initializing..."
color = (255, 255, 255)
current_result = None

alarm_on = False
alarm_thread = None

def sound_alarm():
    global alarm_on
    while alarm_on:
        winsound.Beep(2500, 1000) # Frequency 2500Hz, Duration 1000ms
        if not alarm_on:
            break



# Callback for async processing
def print_result(result: vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global current_result
    current_result = result

# Initialize FaceLandmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=print_result)
detector = vision.FaceLandmarker.create_from_options(options)

# Indices for eyes (based on canonical face mesh)
# These indices might need slight adjustment compared to the legacy solution, 
# but usually they are consistent for 468 landmarks.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def compute_ear(landmarks, indices, w, h):
    # Retrieve landmarks
    # Landmarks in FaceLandmarkerResult are normalized [0,1]
    pts = [landmarks[i] for i in indices]
    
    # Convert to pixel coordinates
    pts_px = [np.array([p.x * w, p.y * h]) for p in pts]
    
    # Vertical distances
    d_v1 = np.linalg.norm(pts_px[1] - pts_px[5])
    d_v2 = np.linalg.norm(pts_px[2] - pts_px[4])
    
    # Horizontal distance
    d_h = np.linalg.norm(pts_px[0] - pts_px[3])
    
    if d_h == 0: return 0.0
    
    ear = (d_v1 + d_v2) / (2.0 * d_h)
    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return ear

def get_head_pose(landmarks, split_w, split_h):
    # 2D image points. If you change the image size, you need to change the vector values
    image_points = np.array([
        (landmarks[1].x * split_w, landmarks[1].y * split_h),     # Nose tip
        (landmarks[152].x * split_w, landmarks[152].y * split_h), # Chin
        (landmarks[33].x * split_w, landmarks[33].y * split_h),   # Left eye left corner
        (landmarks[263].x * split_w, landmarks[263].y * split_h), # Right eye right corner
        (landmarks[61].x * split_w, landmarks[61].y * split_h),   # Left Mouth corner
        (landmarks[291].x * split_w, landmarks[291].y * split_h)  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Camera internals
    focal_length = split_w
    center = (split_w / 2, split_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix
    rmat, jac = cv2.Rodrigues(rotation_vector)

    # Calculate Euler angles
    # We use the rotation matrix to calculate pitch, yaw, roll
    # This formula depends on the camera coordinate system conventions
    # Pitch: Rotation around X-axis
    
    # A cleaner way to get angles is decomposing the projection matrix, but simple Euler extraction works for head pose
    # proj_matrix = np.hstack((rmat, translation_vector))
    # euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 
    
    # Custom Euler calc
    sy = np.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
    singular = sy < 1e-6
    if  not singular :
        x = np.arctan2(rmat[2,1] , rmat[2,2])
        y = np.arctan2(-rmat[2,0], sy)
        z = np.arctan2(rmat[1,0], rmat[0,0])
    else :
        x = np.arctan2(-rmat[1,2], rmat[1,1])
        y = np.arctan2(-rmat[2,0], sy)
        z = 0

    return np.array([x, y, z]) # Returns radians: Pitch, Yaw, Roll


current_cam = 0
cap = cv2.VideoCapture(current_cam)
start_time = time.time()

# Create a named window that can be resized
cv2.namedWindow('Drowsiness Detection (MediaPipe Tasks)', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        # If camera fails to read, try to switch to 0 or just wait
        print(f"Failed to read from camera {current_cam}. Retrying...")
        cap.release()
        current_cam = 0
        cap = cv2.VideoCapture(current_cam)
        time.sleep(1)
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to MP Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Calculate timestamp in ms
    timestamp_ms = int((time.time() - start_time) * 1000)
    
    # Detect asynchronously
    detector.detect_async(mp_image, timestamp_ms)
    
    # Process result if available
    if current_result and current_result.face_landmarks:
        face_landmarks = current_result.face_landmarks[0]
        
        l_ear = compute_ear(face_landmarks, LEFT_EYE, w, h)
        r_ear = compute_ear(face_landmarks, RIGHT_EYE, w, h)
        
        avg_ear = (l_ear + r_ear) / 2.0
        
        # Head Pose Calculation
        pitch, yaw, roll = get_head_pose(face_landmarks, w, h)
        pitch_deg = pitch * 180 / np.pi
        
        
        # State Classification based on EAR
        l_state = 0 # 0=sleep, 1=drowsy, 2=active
        if l_ear > 0.25: l_state = 2
        elif l_ear > 0.21: l_state = 1
        else: l_state = 0
        
        r_state = 0
        if r_ear > 0.25: r_state = 2
        elif r_ear > 0.21: r_state = 1
        else: r_state = 0

        # Check Head Drop
        head_dropped = False
        if pitch_deg < -10: 
             head_dropped = True

        # Counters and Logic
        if l_state == 0 or r_state == 0: # SLEEPING
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (0, 0, 255)
                if not alarm_on:
                    alarm_on = True
                    alarm_thread = threading.Thread(target=sound_alarm)
                    alarm_thread.daemon = True
                    alarm_thread.start()
        
        elif l_state == 2 and r_state == 2: # ACTIVE
            # If head is dropped, we might still consider it NOT fully active?
            # User said: "stop only when eyes open".
            # If I look down with eyes open, alarm should stop?
            # Yes, "as open wela thiyenakota" -> Eyes Open.
            
            sleep = 0
            drowsy = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                if alarm_on:
                    alarm_on = False

        else: # DROWSY (State 1)
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 255, 255)
            # Do NOT stop alarm here. It acts as a latch.

        # Override Status text if Head Dropped (for visual feedback)
        if head_dropped:
             if alarm_on:
                 status = "HEAD DROPPED (ALARM) !"
                 color = (0, 0, 255)
             else:
                 status = "Head Dropped"
                 # color keeps previous (e.g. active green or drowsy yellow)
        
        # Draw text
        cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Pitch: {int(pitch_deg)}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw landmarks
        for idx in LEFT_EYE + RIGHT_EYE:
            lm = face_landmarks[idx]
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (255, 255, 255), -1)

    # Show Camera Index
    cv2.putText(frame, f"Cam: {current_cam} (Press C to switch)", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Drowsiness Detection (MediaPipe Tasks)', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('c') or key == ord('C'):
        print("Switching camera...")
        cap.release()
        current_cam += 1
        # Try to open next camera
        cap = cv2.VideoCapture(current_cam)
        if not cap.isOpened() or not cap.read()[0]:
             print(f"Camera {current_cam} not available. Switching back to 0.")
             current_cam = 0
             cap = cv2.VideoCapture(current_cam)

detector.close()
cap.release()
cv2.destroyAllWindows()
