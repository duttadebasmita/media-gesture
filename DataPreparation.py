import cv2
import mediapipe as mp
import numpy as np
import os

class HandGestureDataCollector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.drawing = mp.solutions.drawing_utils
        self.hands = mp.solutions.hands.Hands(max_num_hands=2)

        self.gesture_labels = {
            'play_pause': 0,
            'next_track': 1,
            'previous_track': 2,
            'volume_up': 3,
            'volume_down': 4,
            'instrument_change': 5,
            'tempo_up': 6,
            'tempo_down': 7,
            'new_track': 8,
            'mute': 9,
            'solo': 10
        }

        self.data_dir = "gesture_data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def capture_gesture(self, gesture_name, num_samples=100):
        assert gesture_name in self.gesture_labels, "Invalid gesture name!"

        print(f"Capturing {num_samples} samples for gesture '{gesture_name}'...")

        sample_count = 0
        while sample_count < num_samples:
            _, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                sample_count += 1
                image_path = os.path.join(self.data_dir, f"{gesture_name}_{sample_count}.jpg")
                cv2.imwrite(image_path, frame)

            cv2.imshow("Gesture Capture", frame)
            if cv2.waitKey(1) == 27:
                break

        print(f"Captured {sample_count} samples for gesture '{gesture_name}'.")

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = HandGestureDataCollector()
    
    # Example: Capture 100 samples for each gesture
    gestures = ['play_pause', 'next_track', 'previous_track', 'volume_up', 'volume_down', 
                'instrument_change', 'tempo_up', 'tempo_down', 'new_track', 'mute', 'solo']

    for gesture in gestures:
        collector.capture_gesture(gesture, num_samples=100)
    
    collector.close()
