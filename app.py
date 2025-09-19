from flask import Flask, render_template, Response
import cv2
import time

# Flask app initialization
app = Flask(__name__)

# Model configuration
model_path = 'face_recognizer.yml'  
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Labels and tracking variables
label_names = ['Normal', 'Suspicious']
suspicious_detected = False
start_time = None
suspicious_duration = 0
suspicious_occurrences = 0  # Counter for suspicious activity

def generate_frames():
    global suspicious_occurrences, suspicious_detected, start_time, suspicious_duration
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            label, confidence = recognizer.predict(face)

            color = (0, 255, 0) if label == 0 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{label_names[label]} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Logic for suspicious activity detection
            if label == 1:  # Suspicious activity detected
                if not suspicious_detected:
                    suspicious_detected = True
                    start_time = time.time()
                    suspicious_occurrences += 1  # Increment counter on new detection
                else:
                    suspicious_duration = time.time() - start_time

                duration_text = f"Duration: {suspicious_duration:.2f}s"
                cv2.putText(frame, duration_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Display occurrences below the red rectangle
                occurrences_text = f"Occurrences: {suspicious_occurrences}"
                cv2.putText(frame, occurrences_text, (x, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:  # Normal state
                if suspicious_detected:
                    suspicious_detected = False
                    start_time = None

        # Display occurrences of suspicious activity on the top
        occurrences_text = f"Suspicious Occurrences: {suspicious_occurrences}"
        cv2.putText(frame, occurrences_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode frame and send it as a response
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
