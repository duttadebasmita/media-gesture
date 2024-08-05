import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the data
print("Loading data...")
data = np.load('data.npy')
labels = np.load('labels.npy')

# Display the shape of the loaded data
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Standardize the data
print("Standardizing data...")
scaler = StandardScaler()
data = scaler.fit_transform(data)

# One-hot encode labels
print("One-hot encoding labels...")
labels = to_categorical(labels - 1)  # Assuming labels start from 1

# Split the data
print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Save the preprocessed data
print("Saving preprocessed data...")
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# Confirming the saved files
print("Preprocessed data saved successfully.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
print("Scaler mean and scale saved.")
