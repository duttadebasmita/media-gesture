import os
import numpy as np
import cv2

def create_dataset(data_dir):
    data = []
    labels = []

    gesture_labels = {
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

    for gesture, label in gesture_labels.items():
        gesture_files = [f for f in os.listdir(data_dir) if f.startswith(gesture)]
        
        for file in gesture_files:
            img_path = os.path.join(data_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (128, 128))  # Resize images to a consistent size
            data.append(img)
            labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    np.save('data.npy', data)
    np.save('labels.npy', labels)

if __name__ == "__main__":
    create_dataset("gesture_data")
