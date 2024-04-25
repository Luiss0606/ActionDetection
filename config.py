import mediapipe as mp
import os

class GlobalVars:
    MP_HOLISTIC = mp.solutions.holistic # Holistic model
    MP_DRAWING = mp.solutions.drawing_utils # Drawing utilities

    # Path for exported data, numpy arrays
    DATA_PATH = os.path.join('MP_Data') 

    # Actions that we try to detect
    ACTIONS = ['hello', 'thanks', 'iloveyou']

    # Thirty videos worth of data
    NO_SEQUENCES = 30

    # Videos are going to be 30 frames in length
    SEQUENCE_LENGTH = 30

    # Folder start
    START_FOLDER = 0

    COLORS = [(245,117,16), (117,245,16), (16,117,245)]
