import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui
import time

class HandGestureController:
    def __init__(self):
        self.drawing = mp.solutions.drawing_utils
        self.hands = mp.solutions.hands.Hands(max_num_hands=2)

        self.start_init = False
        self.prev_right = -1
        self.prev_left = -1

        self.check_permissions()
        self.ensure_app_compatibility()
        self.fine_tune_gesture_recognition()

        self.model = tf.keras.models.load_model('gesture_model.h5')
        self.scaler_mean = np.load('scaler_mean.npy')
        self.scaler_scale = np.load('scaler_scale.npy')

    def preprocess_landmarks(self, hand_landmarks):
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        landmarks = landmarks.flatten()

        expected_shape = self.scaler_mean.shape[0]
        if landmarks.shape[0] < expected_shape:
            landmarks = np.pad(landmarks, (0, expected_shape - landmarks.shape[0]), 'constant')
        elif landmarks.shape[0] > expected_shape:
            landmarks = landmarks[:expected_shape]

        landmarks = (landmarks - self.scaler_mean) / self.scaler_scale
        return landmarks

    def predict_gesture(self, hand_landmarks):
        landmarks = self.preprocess_landmarks(hand_landmarks)
        prediction = self.model.predict(np.array([landmarks]))
        gesture = np.argmax(prediction, axis=1)[0]
        return gesture, prediction

    def count_fingers(self, hand_landmarks):
        fingers = []
        if hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].x:
            fingers.append(1)
        else:
            fingers.append(0)

        for tip in [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp.solutions.hands.HandLandmark.RING_FINGER_TIP, mp.solutions.hands.HandLandmark.PINKY_TIP]:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

    def check_permissions(self):
        try:
            pyautogui.press('volumedown')
        except Exception as e:
            st.error("Permission issue detected with pyautogui. Please ensure it has necessary permissions.")
            st.stop()

    def ensure_app_compatibility(self):
        supported_apps = {
            'Ableton Live': 'https://www.ableton.com/',
            'FL Studio': 'https://www.image-line.com/flstudio/',
            'Logic Pro': 'https://www.apple.com/logic-pro/',
            'GarageBand': 'https://www.apple.com/mac/garageband/',
            'Pro Tools': 'https://www.avid.com/pro-tools',
            'Zoom': 'https://zoom.us/',
            'Microsoft Teams': 'https://www.microsoft.com/microsoft-teams/',
            'Google Meet': 'https://meet.google.com/',
            'VLC Media Player': 'https://www.videolan.org/vlc/',
            'Spotify': 'https://www.spotify.com/',
            'iTunes/Apple Music': 'https://www.apple.com/apple-music/'
        }

        st.sidebar.write("Ensure your application is one of the following or supports keyboard shortcuts:")
        for app, url in supported_apps.items():
            st.sidebar.write(f"[{app}]({url})")

    def fine_tune_gesture_recognition(self):
        st.sidebar.write("Adjusting gesture recognition thresholds if necessary...")

    def run(self):
        stframe = st.empty()
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                st.error("Error: Failed to capture frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                hand_keypoints = results.multi_hand_landmarks

                right_fingers, left_fingers = -1, -1

                if len(hand_keypoints) > 0:
                    right_hand = hand_keypoints[0]
                    right_fingers = self.count_fingers(right_hand)
                    gesture, prediction = self.predict_gesture(right_hand)
                    st.sidebar.write(f"Right Hand Gesture: {gesture} with confidence scores: {prediction}")

                if len(hand_keypoints) > 1:
                    left_hand = hand_keypoints[1]
                    left_fingers = self.count_fingers(left_hand)
                    gesture, prediction = self.predict_gesture(left_hand)
                    st.sidebar.write(f"Left Hand Gesture: {gesture} with confidence scores: {prediction}")

                if right_fingers != self.prev_right or left_fingers != self.prev_left:
                    if not self.start_init:
                        self.start_time = time.time()
                        self.start_init = True
                    elif (time.time() - self.start_time) > 0.2:
                        self.perform_action(right_fingers, left_fingers)
                        self.prev_right = right_fingers
                        self.prev_left = left_fingers
                        self.start_init = False

                for hand_landmarks in hand_keypoints:
                    self.drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            stframe.image(frame, channels="BGR")

        cap.release()

    def perform_action(self, right_fingers, left_fingers):
        action = ""
        if right_fingers == 1:
            action = "Play/Pause"
            pyautogui.press("playpause")
        elif right_fingers == 2:
            action = "Next Track"
            pyautogui.press("nexttrack")
        elif right_fingers == 3:
            action = "Previous Track"
            pyautogui.press("prevtrack")
        elif right_fingers == 4:
            action = "Volume Up"
            pyautogui.press("volumeup")
        elif right_fingers == 5:
            action = "Volume Down"
            pyautogui.press("volumedown")
        elif right_fingers == 5:
            if left_fingers == 1:
                action = "Instrument Change"
                pyautogui.press("instrument_change")
            elif left_fingers == 2:
                action = "Tempo Up"
                pyautogui.press("tempo_up")
            elif left_fingers == 3:
                action = "Tempo Down"
                pyautogui.press("tempo_down")
            elif left_fingers == 4:
                action = "New Track"
                pyautogui.press("new_track")
            elif left_fingers == 5:
                action = "Mute"
                pyautogui.press("mute")
            elif right_fingers == 4 and left_fingers == 4:
                action = "Solo"
                pyautogui.press("solo")
            elif right_fingers == 1 and left_fingers == 2:
                action = "Fast Forward"
                pyautogui.hotkey('shift', 'right')
            elif right_fingers == 1 and left_fingers == 3:
                action = "Rewind"
                pyautogui.hotkey('shift', 'left')
        if action:
            st.sidebar.write(f"Performing action: {action}")

if __name__ == "__main__":
    st.title("Hand Gesture Control")
    st.write("Control your applications with hand gestures.")
    controller = HandGestureController()
    controller.run()
