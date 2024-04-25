import cv2
import numpy as np
import os
import mediapipe as mp

from functions import mediapipe_detection, draw_styled_landmarks, extract_keypoints

from config import GlobalVars

# Show the current working directory
os.getcwd()
# Create folders for each action
for action in GlobalVars.ACTIONS:
    for num, seq in enumerate(range(GlobalVars.NO_SEQUENCES)):
        try:
            os.makedirs(os.path.join(GlobalVars.DATA_PATH, action, str(seq)))
        except FileExistsError:
            print(f'Folder {action}, {seq} already exists')
    


# Set mediapipe model
cap = cv2.VideoCapture(0)

with GlobalVars.MP_HOLISTIC.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop through ACTIONS
    for action in GlobalVars.ACTIONS:
        # Loop through sequences aka videos
        for sequence in range(GlobalVars.START_FOLDER, GlobalVars.START_FOLDER + GlobalVars.NO_SEQUENCES):
            # Loop through video length aka sequence length
            for frame_num in range(GlobalVars.SEQUENCE_LENGTH):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Apply wait logic
                if frame_num == 0:
                    cv2.putText(
                        image,
                        "STARTING COLLECTION",
                        (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        "Collecting frames for {} Video Number {}".format(
                            action, sequence
                        ),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # Show to screen
                    cv2.imshow("OpenCV Feed", image)
                    cv2.waitKey(500)
                else:
                    cv2.putText(
                        image,
                        "Collecting frames for {} Video Number {}".format(
                            action, sequence
                        ),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # Show to screen
                    cv2.imshow("OpenCV Feed", image)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(
                    GlobalVars.DATA_PATH, action, str(sequence), str(frame_num) + '.npy'
                )
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()