# Driver Drowsiness Detection (MediaPipe Version)

This project implements a driver drowsiness detection system using **Google MediaPipe** and **OpenCV**. It monitors eye blinking and head posture to detect fatigue and alerts the driver using an alarm.

## Features

- **MediaPipe Face Mesh**: Uses high-precision face landmarker models instead of the heavy Dlib library.
- **Real-time Drowsiness Detection**: Monitors Eye Aspect Ratio (EAR) to detect sleeping/closed eyes.
- **Head Drop Detection**: Monitors mental fatigue by checking if the head drops (Pitch angle).
- **Audio Alarm**: Triggers a loud beep sound when sleeping is detected.
  - _Note_: The alarm continues if you are sleeping, even if your head drops. It only stops when you become "Active" (eyes open).
- **Camera Switching**: Press **'C'** to switch between available cameras.
- **Responsive UI**: The window can be resized to fit any screen.

## Requirements

- Python 3.10+
- `opencv-python`
- `mediapipe`
- `numpy`

## Installation

1.  Install dependencies:
    ```bash
    pip install opencv-python mediapipe numpy
    ```
2.  The model file `face_landmarker.task` will be automatically downloaded on the first run if not present.

## How to Run

1.  Open your terminal or command prompt in the project folder.
2.  Run the following command:

    **Windows:**

    ```bash
    py drowsiness_detect_mediapipe.py
    ```

    **Mac/Linux:**

    ```bash
    python3 drowsiness_detect_mediapipe.py
    ```

### Controls

- **ESC**: Exit the application.
- **C**: Switch Camera (if multiple cameras are connected).

## Logic

The system calculates the **Eye Aspect Ratio (EAR)** to determine the state:

- **Active :)** (Green): Eyes Open (EAR > 0.25)
- **Drowsy !** (Yellow): Eyes partially closed (0.21 < EAR < 0.25)
- **SLEEPING !!!** (Red): Eyes Closed (EAR < 0.21) -> **ALARM TRIGGERS**

It also calculates **Head Pitch**:

- If Pitch < -10 degrees: **Head Dropped** detected.

## Credits

Built using MediaPipe Tasks API and OpenCV.
