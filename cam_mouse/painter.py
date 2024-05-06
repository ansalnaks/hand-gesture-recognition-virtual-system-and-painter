import cv2
import numpy as np
import mediapipe as mp

def draw_on_canvas(canvas, point):
    cv2.circle(canvas, point, 5, (255, 255, 255), -1)  # Adjust line width here

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a blank canvas
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Initialize mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Set the position of "Press C" text
    text_position = (50, 50)
    text_color = (255, 255, 255)
    text_thickness = 2
    cv2.putText(canvas, 'Press C to Clear', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, text_thickness)

    while True:
        # Read frame from video capture
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from video capture.")
            break
        frame = cv2.flip(frame, 1)

        # Detect hand landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Draw hand landmarks and fingertips
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger tip coordinates
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_finger_tip_px = int(index_finger_tip.x * width), int(index_finger_tip.y * height)

                # Get finger positions
                fingers = [hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                           hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                           hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                           hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                           hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y]

                # Draw on canvas only if the index finger is up and all other fingers are down
                if fingers[1] < min(fingers[:1] + fingers[2:]):
                    draw_on_canvas(canvas, index_finger_tip_px)

        # Display frame and canvas
        cv2.imshow("Frame", frame)
        cv2.imshow("Canvas", canvas)

        # Check for keyboard events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):  # Clear canvas if 'c' is pressed
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(canvas, 'Press C to Clear', text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, text_thickness)

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
