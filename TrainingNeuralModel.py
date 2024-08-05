import numpy as np
import pandas as pd  # Add this line
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Set random seed for reproducibility
RANDOM_SEED = 42

# Load data
data = np.load('data.npy')
labels = np.load('labels.npy')

# Inspect the shape of data and labels
print(f"Original data shape: {data.shape}")
print(f"Original labels shape: {labels.shape}")

# Reshape data if necessary
if len(data.shape) > 2:
    # Flatten each sample to be 2D (samples, features)
    data = data.reshape(data.shape[0], -1)
    print(f"Reshaped data to: {data.shape}")

# Encode labels if not already encoded
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Standardize the data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# One-hot encode labels
labels = to_categorical(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=RANDOM_SEED)

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(y_train.shape[1], activation='softmax')
])

model.summary()

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Save the model in HDF5 format
model.save('gesture_model.h5')

# Save the scaler parameters
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)

# Confusion matrix and classification report
def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(np.argmax(y_true, axis=1))))
    cmx_data = confusion_matrix(np.argmax(y_true, axis=1), y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False)
    ax.set_ylim(len(set(np.argmax(y_true, axis=1))), 0)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(np.argmax(y_true, axis=1), y_pred))
        
        precision = precision_score(np.argmax(y_true, axis=1), y_pred, average='weighted')
        recall = recall_score(np.argmax(y_true, axis=1), y_pred, average='weighted')
        f1 = f1_score(np.argmax(y_true, axis=1), y_pred, average='weighted')
        
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

print_confusion_matrix(y_test, y_pred)

# Save the label encoder classes
np.save('label_classes.npy', label_encoder.classes_)
