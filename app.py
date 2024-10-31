# from flask import Flask,render_template,request
# import os
# from fastai.vision.all import *
# app=Flask(__name__)

# path=os.getcwd() 
# imagetest="test"
# modelpath=os.path.join(path,'fruit_classifier.pkl')

# model=load_learner(modelpath)
# @app.route('/')
# def home():
#     title="WELCOME TO OUR FRUIT RECOGNIZER APP"
#     title={"tname":title}
#     return render_template('home.html',title=title)

# @app.route('/prediction/')    
# def show_prediction():
#     imagename=request.args.get('filename')
#     fullpath=os.path.join(path,imagetest,imagename)
#     image=PILImage.create(fullpath)
#     label,_,prob=model.predict(image)
#     predstring=f"The fruit is: {label}" 
#     prediction={"pred":predstring}
#     return render_template('show_prediction.html',prediction=prediction)


# if __name__=="__main__":
#     app.run(debug=True)    



     
from flask import Flask, render_template, request,jsonify
import os
import pickle 
import cv2
import numpy as np
from fastai.vision.all import *
from tensorflow.keras.models import load_model
import base64
app = Flask(__name__)

# Define the path for the model
path = os.getcwd()
modelpath = os.path.join(path, 'fruit_classifier.pkl')
emotion_modelpath = os.path.join(path, 'emotion_model.h5')  # Path for the emotion detection model

model = load_learner(modelpath)

emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


emotion_model = load_model('emotion_model.h5')

print(type(emotion_model))    
@app.route("/")
def home():
    return render_template('home.html')

@app.route('/fruit-recognizer')
def fruit():
    title = "WELCOME TO OUR FRUIT RECOGNIZER APP"
    title = {"tname": title}
    return render_template('fruit-recognizer.html', title=title)

@app.route('/prediction/', methods=['POST'])
def show_prediction():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    try:
        # Use PILImage.create to process the uploaded image directly from the file stream
        image = PILImage.create(file.stream)
        
        # Make prediction
        label, _, prob = model.predict(image)
        predstring = f"The fruit is: {label}" 
        prediction = {"pred": predstring}
        
        return render_template('show_prediction.html', prediction=prediction)

    except Exception as e:
        return str(e)  # Return the error message for debugging
    



@app.route('/emotion-detect')
def emotion_detect():
    return render_template('emotion_detect.html')

def predict_emotion(frame):
    # Preprocess the frame for emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))  # Resize to model input size
        face_roi = face_roi / 255.0  # Normalize to [0, 1]
        face_roi = face_roi.reshape(1, 48, 48, 1)  # Reshape for model input

        # Make prediction
        prediction = emotion_model.predict(face_roi)
        emotion_index = np.argmax(prediction)  # Get the index of the highest probability
        emotion_label = emotion_labels[emotion_index]  # Map index to emotion label

        # Store the results
        results.append({"label": emotion_label, "coordinates": (x, y, w, h)})

    return results

# @app.route('/capture')
# def capture():
#     # Start capturing video from the webcam
#     cap = cv2.VideoCapture(0)
#     frame_count = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Predict emotions in the current frame
#         results = predict_emotion(frame)

#         # Draw results on the frame
#         for detection in results:
#             label = detection["label"]
#             x, y, w, h = detection["coordinates"]

#             # Draw rectangle around detected face and label
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#             cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#         # Display the frame with detected emotions
#         cv2.imshow('Emotion Detection', frame)

#         # Exit loop on pressing 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         frame_count += 1

#     cap.release()
#     cv2.destroyAllWindows()
#     return jsonify({"status": "capturing stopped"})

@app.route('/capture')
def capture():
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)
    frame_count = 0
    results_data = []  # List to store results for each frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict emotions in the current frame
        results = predict_emotion(frame)

        # Convert results to JSON-serializable format
        serializable_results = []
        for detection in results:
            # Ensure that all elements are JSON serializable
            serializable_detection = {
                "label": detection["label"],
                "coordinates": [int(coord) for coord in detection["coordinates"]]  # Convert to int
            }
            serializable_results.append(serializable_detection)

        results_data.append(serializable_results)  # Store results for the current frame

        # Draw results on the frame
        for detection in serializable_results:
            label = detection["label"]
            x, y, w, h = detection["coordinates"]

            # Draw rectangle around detected face and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Display the frame with detected emotions
        cv2.imshow('Emotion Detection', frame)

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Return the captured results as JSON
    return jsonify({"status": "capturing stopped", "results": results_data[-1]})


@app.route("/motion_detection")
def detc():
      return render_template("motion_detection.html")

@app.route("/segmentation")
def det():
      return render_template("semantic.html")

@app.route("/yolo")
def yolo():
      return render_template("yolo.html")

@app.route("/face-recognition")
def fr():
      return render_template("face-recognition.html")







if __name__ == "__main__":
    app.run(debug=True)





if __name__ == "__main__":
    app.run(debug=True)