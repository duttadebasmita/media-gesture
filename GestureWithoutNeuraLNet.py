import cv2
import mediapipe as mp
import pyautogui
import time
import sys

class HandGestureController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.drawing = mp.solutions.drawing_utils
        self.hands = mp.solutions.hands.Hands(max_num_hands=2)

        self.start_init = False
        self.prev_right = -1
        self.prev_left = -1

        self.check_permissions()
        self.ensure_app_compatibility()
        self.fine_tune_gesture_recognition()

    def count_fingers(self, hand_landmarks):
        """
        Count the number of extended fingers for a given hand.
        """
        fingers = []

        # Thumb
        if hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
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
            print("Permission issue detected with pyautogui. Please ensure it has necessary permissions.")
            sys.exit()

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

        print("Ensure your application is one of the following or supports keyboard shortcuts:")
        for app, url in supported_apps.items():
            print(f"{app}: {url}")

    def fine_tune_gesture_recognition(self):
        print("Adjusting gesture recognition thresholds if necessary...")

    def run(self):
        while True:
            end_time = time.time()
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                hand_keypoints = results.multi_hand_landmarks

                right_fingers, left_fingers = -1, -1

                if len(hand_keypoints) > 0:
                    right_hand = hand_keypoints[0]
                    right_fingers = self.count_fingers(right_hand)

                if len(hand_keypoints) > 1:
                    left_hand = hand_keypoints[1]
                    left_fingers = self.count_fingers(left_hand)

                if right_fingers != self.prev_right or left_fingers != self.prev_left:
                    if not self.start_init:
                        self.start_time = time.time()
                        self.start_init = True
                    elif (end_time - self.start_time) > 0.2:
                        self.perform_action(right_fingers, left_fingers)
                        self.prev_right = right_fingers
                        self.prev_left = left_fingers
                        self.start_init = False

                for hand_landmarks in hand_keypoints:
                    self.drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            cv2.imshow("Hand Gesture Control", frame)

            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                self.cap.release()
                break

    def perform_action(self, right_fingers, left_fingers):
        print(f"Performing action for right fingers: {right_fingers}, left fingers: {left_fingers}")

        if right_fingers == 1:
            print("Action: Play/Pause")
            pyautogui.press("playpause")
        elif right_fingers == 2:
            print("Action: Next Track")
            pyautogui.press("nexttrack")
        elif right_fingers == 3:
            print("Action: Previous Track")
            pyautogui.press("prevtrack")
        elif right_fingers == 4:
            print("Action: Volume Up")
            pyautogui.press("volumeup")
        elif right_fingers == 5:
            print("Action: Volume Down")
            pyautogui.press("volumedown")
        elif right_fingers == 5:
            if left_fingers == 1:
                print("Action: Instrument Change")
                pyautogui.press("instrument_change")
            elif left_fingers == 2:
                print("Action: Tempo Up")
                pyautogui.press("tempo_up")
            elif left_fingers == 3:
                print("Action: Tempo Down")
                pyautogui.press("tempo_down")
            elif left_fingers == 4:
                print("Action: New Track")
                pyautogui.press("new_track")
            elif left_fingers == 5:
                print("Action: Mute")
                pyautogui.press("mute")
        elif right_fingers == 4 and left_fingers == 4:
            print("Action: Solo")
            pyautogui.press("solo")

if __name__ == "__main__":
    controller = HandGestureController()
    controller.run()
