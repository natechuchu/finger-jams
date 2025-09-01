import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=2, detection_conf=0.5, tracking_conf=0.5):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.drawing = mp.solutions.drawing_utils
        self.drawing_styles = mp.solutions.drawing_styles

    def process(self, image: np.ndarray):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        return results

    def draw_hands(self, image: np.ndarray, results): # What is results
        #bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    self.drawing_styles.get_default_hand_landmarks_style(),
                    self.drawing_styles.get_default_hand_connections_style()
                )
        return image