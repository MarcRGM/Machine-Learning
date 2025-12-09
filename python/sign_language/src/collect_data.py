import cv2 # webcam
import mediapipe as mp # Detects hand and gives it Landmark

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand detection
hands = mp_hands.Hands (
    static_image_mode=False # Track hands across frames
    max_num_hands=2 # detects 1 hand
    min_detection_confidence=0.5 # 50% hand detection
    min_tracking_confidence=0.5 # 50% tracking between frames
)

# Opens the webcam
cap = cv2.VideoCapture(0) # open camera index 0 (Main camera)
# cap read frames from camera



