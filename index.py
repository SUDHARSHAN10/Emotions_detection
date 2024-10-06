from flask import Flask, flash, redirect, render_template, request, url_for, Response
import operator
from keras import backend as K
import cv2
import random
import os
import sys
from keras.models import load_model
import numpy as np
from gtts import gTTS
from inference import detect_faces
from inference import draw_text
from inference import draw_bounding_box
from inference import apply_offsets
from inference import load_detection_model
from inference import load_image
from preprocessor import preprocess_input

# from fer import FER
app = Flask(__name__)
# camera_port = 0
# camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
#camera = cv2.VideoCapture(0)

a = None

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def gen_frames():
    global camera
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        success, frame = camera.read()  # read the camera frame
        cv2.imwrite('testingimg.jpg', frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/cam')
def cam():

    return render_template('cam.html')

@app.route('/capture')
def capture():
    camera.release()

    return redirect(url_for('stories'))


    #return render_template('cam.html')


@app.route('/stories')
def stories():
    # camera.release()
    s = []
    q = None

    K.clear_session()

    image_path = "testingimg.jpg"
    detection_model_path = '../Emotion_Stories/models/haarcascade_frontalface_default.xml'
    emotion_model_path = '../Emotion_Stories/models/cnn_model.hdf5'
    emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}
    emotion_text = ""

    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    face_detection = load_detection_model(detection_model_path)
    emotion_classifier = load_model(emotion_model_path, compile=False)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []
    print("ip=", image_path)
    img = cv2.imread(image_path)
    print("img=", img)

    bgr_image = img
    print("imge=", bgr_image)
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    for face_coordinates in faces:
        print("face")

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]

    emo = emotion_text

    if emo=="":
        return render_template('index.html', msg="failed")
    else:
        K.clear_session()

        print("emo=", emo)

        from stories import angry_stories, happy_stories, sad_stories, fear_stories, neutral_stories

        aquote = angry_stories()
        hquote = happy_stories()
        squote = sad_stories()
        fquote = fear_stories()
        nquote = neutral_stories()

        if emo == 'angry':
            q = random.sample(aquote, 1)
        elif emo == 'neutral':
            q = random.sample(nquote, 1)
        elif emo == 'happy':
            q = random.sample(hquote, 1)
        elif emo == 'sad':
            q = random.sample(squote, 1)
        elif emo == 'fear':
            q = random.sample(fquote, 1)

        obj = gTTS(text=q[0], lang='en', slow=False)
        obj.save("../Emotion_Stories/static/story.mp3")

        return render_template('stories.html', emotion=emo, quote=q[0])


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host="localhost", port="2024", debug=True)