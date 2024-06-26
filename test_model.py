import os
import cv2
import numpy as np

from config import GlobalVars
from functions import (
    extract_keypoints,
    draw_styled_landmarks,
    mediapipe_detection,
    prob_viz,
)
from tensorflow.keras.models import load_model


# Dowload the last model
# with open("last_model.txt", "r") as f:
#     last_model = f.read()
#     model = load_model(last_model)
model = load_model('action.h5')

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

cap = cv2.VideoCapture(0)
# Set mediapipe model
with GlobalVars.MP_HOLISTIC.Holistic(
    min_detection_confidence=0.5, min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(GlobalVars.ACTIONS[np.argmax(res)])
            predictions.append(np.argmax(res))

            # 3. Viz logic
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:

                    if len(sentence) > 0:
                        if GlobalVars.ACTIONS[np.argmax(res)] != sentence[-1]:
                            sentence.append(GlobalVars.ACTIONS[np.argmax(res)])
                    else:
                        sentence.append(GlobalVars.ACTIONS[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, GlobalVars.ACTIONS, image, GlobalVars.COLORS)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(
            image,
            " ".join(sentence),
            (3, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Show to screen
        cv2.imshow("OpenCV Feed", image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
