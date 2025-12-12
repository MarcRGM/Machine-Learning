import cv2 # webcam
import mediapipe as mp # Detects hand and gives it Landmark

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Hand detection
hands = mp_hands.Hands(
    static_image_mode=False, # Track hands across frames
    max_num_hands=2, # detects 1 hand
    min_detection_confidence=0.5, # 50% hand detection
    min_tracking_confidence=0.5 # 50% tracking between frames
)

# Opens the webcam
cap = cv2.VideoCapture(0) # open camera index 0 (Main camera)
# cap read frames from camera

# Runs every frame
while True:
    success, frame = cap.read() # Tries to read one frame from the webcam
    # success is True if it got a frame, False if it failed
    # frame is a NumPy array representing the image
    if not success:
        break
    
    # Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    # OpenCV frames are in BGR (blue, green, red) order
    # MediaPipe expects RGB (red, green, blue)
    # cv2.cvtColor converts the frame from BGR to RGB and stores it in rgb

    result = hands.process(rgb) # passes the image to the MediaPipe model
    # It returns a result object like:
    # result.multi_hand_landmarks – list of detected hands, each with 21 landmarks

    if result.multi_hand_landmarks: # If this is not empty, then at least one hand was detected
        for hand_landmarks in result.multi_hand_landmarks: # Loop over each detected hand
            # hand_landmarks is a structure with 21 points
            mp_drawing.draw_landmarks( # Drawing landmarks on the video
                frame, # what to draw on
                hand_landmarks, # which landmarks to draw
                mp_hands.HAND_CONNECTIONS # how the points are connected
            )

            # hand_landmarks.landmark is a list of 21 landmarks
            # each landmark contains x, y, and z coordinates
            # Example: 
            points = []
            for lm in hand_landmarks.landmark:
                points.append(lm.x)
                points.append(lm.y)
                points.append(lm.z)

            print("Number of values:", len(points)) 
            # 21 landmarks * 3 coordinates each = 63 values

    
    # Shows the current frame in a window named "Collect Data - Press q to quit"
    cv2.imshow("Collect Data - Press q to quit", frame)

    key = cv2.waitKey(1) & 0xFF
    # cv2.waitKey(1) waits 1 millisecond for a key press
    # & 0xFF is a common OpenCV pattern to get the key code

    if key == ord('q'):
        break
    # ord('q') is the keycode for the letter q
    # Pressing q will break the loop and the program goes to cleanup

    elif key == ord('a'):
        print("Captured an 'A' sample with 63 values")
    else:
        print("No valid hand data to save")
    # This is for capturing the points 
    
    

cap.release() # closes the webcam
cv2.destroyAllWindows() # closes all OpenCV windows
hands.close() # releases resources used by MediaPipe
