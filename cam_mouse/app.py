from flask import Flask, render_template, request
from controller import GestureManager
import threading
import cv2
import mediapipe as mp
import time
from painter import main as painter_main


app = Flask(__name__, static_folder='static')
manager = GestureManager()
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

stop_signal = False  # Flag variable to stop gesture recognition

def start_gesture_recognition():
    global manager
    global hands
    global cap
    global mp_hands
    global mp_draw
    global stop_signal
    while not stop_signal:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgbFrame)
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 1:
                manager.hand_Landmarks = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, manager.hand_Landmarks, mp_hands.HAND_CONNECTIONS)
            manager.update_fingers_status()
            manager.cursor_moving()
            manager.detect_scrolling()
            manager.detect_clicking()
            manager.detect_dragging()
            manager.detect_zoomming()

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(5) & 0xff == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Gesture recognition stopped.")

    app = Flask(__name__, static_url_path='/static')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_gesture():
    global manager
    threading.Thread(target=start_gesture_recognition).start()
    return 'Gesture recognition started.'

@app.route('/stop', methods=['POST'])
def stop_gesture():
    global stop_signal
    stop_signal = True
    return 'Gesture recognition stopping...'

@app.route('/painter', methods=['POST'])  # Define route for /painter with POST method
def painter():
    if request.method == 'POST':
        painter_main()  # Call the main function from painter.py
        return 'Painter function executed successfully'

    return 'Method not allowed'



@app.route('/mainpage')
def main():
    return render_template('mainpage.html')

@app.route('/left')
def left():
    return render_template('left.html')

@app.route('/right')
def right():
    return render_template('right.html')

@app.route('/double')
def double():
    return render_template('double.html')

@app.route('/scrollup')
def scrollup():
    return render_template('scrollup.html')

@app.route('/scrolldown')
def scrolldown():
    return render_template('scrolldown.html')

@app.route('/zoomin')
def zoomin():
    return render_template('zoomin.html')

@app.route('/zoomout')
def zoomout():
    return render_template('zoomout.html')

@app.route('/drag')
def drag():
    return render_template('drag.html')

if __name__ == '__main__':
    app.run(debug=True)
