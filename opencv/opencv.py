import cv2
import numpy as np
import os
import time
from matplotlib import pyplot as plt
import mediapipe as mp

# holistic model for tracking body joints
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils     # drawing utilities


def mediapipe_detection(image, model):
    # Converts an image from one color space to another
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Making image unwritable before processing to save memory
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# colour formats follow BGR


def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
    #  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))  # white

    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))  # white

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1))    # blue

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))    # green

    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))    # red


cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(frame, results)
        cv2.imshow('OpenCV Feed', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
