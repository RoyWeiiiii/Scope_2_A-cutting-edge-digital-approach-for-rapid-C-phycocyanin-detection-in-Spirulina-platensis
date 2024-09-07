import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow_addons.metrics import RSquare
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data Preparation
main_data_dir = "E:\CodingProjects\machine_learning\Experiment_3\CNN_Biomass_regressor"
days = ['Day_2','Day_4','Day_6','Day_8','Day_10','Day_12']
devices = ['Iphone_13_Pro_Max', 'Nikon_Z50']
conditions = ['Covered', 'Light_disturbance']
batches = ['Batch_1_crop', 'Batch_2_crop', 'Batch_3_crop']

images = []
cpc_concentrations = []

desired_pixel_size = (250, 250)

# Define CPC concentrations for each batch
cpc_concentration_map = {
    ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.0833,
    ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.0760,
    ( 'Day_2', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.0766,
    ( 'Day_2', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_1_crop'): 0.0833,
    ( 'Day_2', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_2_crop'): 0.0760,
    ( 'Day_2', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_3_crop'): 0.0766,
    ( 'Day_2', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.0833,
    ( 'Day_2', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.0760,
    ( 'Day_2', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.0766,
    ( 'Day_2', 'Nikon_Z50','Light_disturbed', 'Batch_1_crop'): 0.0833,
    ( 'Day_2', 'Nikon_Z50','Light_disturbed', 'Batch_2_crop'): 0.0760,
    ( 'Day_2', 'Nikon_Z50','Light_disturbed', 'Batch_3_crop'): 0.0766,
    ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.1538,
    ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.1175,
    ( 'Day_4', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.1734,
    ( 'Day_4', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_1_crop'): 0.1538,
    ( 'Day_4', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_2_crop'): 0.1175,
    ( 'Day_4', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_3_crop'): 0.1734,
    ( 'Day_4', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.1538,
    ( 'Day_4', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.1175,
    ( 'Day_4', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.1734,
    ( 'Day_4', 'Nikon_Z50','Light_disturbed', 'Batch_1_crop'): 0.1538,
    ( 'Day_4', 'Nikon_Z50','Light_disturbed', 'Batch_2_crop'): 0.1175,
    ( 'Day_4', 'Nikon_Z50','Light_disturbed', 'Batch_3_crop'): 0.1734,
    ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.2291,
    ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.2193,
    ( 'Day_6', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.2193,
    ( 'Day_6', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_1_crop'): 0.2291,
    ( 'Day_6', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_2_crop'): 0.2193,
    ( 'Day_6', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_3_crop'): 0.2193,
    ( 'Day_6', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.2291,
    ( 'Day_6', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.2193,
    ( 'Day_6', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.2193,
    ( 'Day_6', 'Nikon_Z50','Light_disturbed', 'Batch_1_crop'): 0.2291,
    ( 'Day_6', 'Nikon_Z50','Light_disturbed', 'Batch_2_crop'): 0.2193,
    ( 'Day_6', 'Nikon_Z50','Light_disturbed', 'Batch_3_crop'): 0.2193,
    ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.4220,
    ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.4458,
    ( 'Day_8', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.4261,
    ( 'Day_8', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_1_crop'): 0.4220,
    ( 'Day_8', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_2_crop'): 0.4458,
    ( 'Day_8', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_3_crop'): 0.4261,
    ( 'Day_8', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.4220,
    ( 'Day_8', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.4458,
    ( 'Day_8', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.4261,
    ( 'Day_8', 'Nikon_Z50','Light_disturbed', 'Batch_1_crop'): 0.4220,
    ( 'Day_8', 'Nikon_Z50','Light_disturbed', 'Batch_2_crop'): 0.4458,
    ( 'Day_8', 'Nikon_Z50','Light_disturbed', 'Batch_3_crop'): 0.4261,
    ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.3856,
    ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.2757,
    ( 'Day_10', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.3795,
    ( 'Day_10', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_1_crop'): 0.3856,
    ( 'Day_10', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_2_crop'): 0.2757,
    ( 'Day_10', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_3_crop'): 0.3795,
    ( 'Day_10', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.3856,
    ( 'Day_10', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.2757,
    ( 'Day_10', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.3795,
    ( 'Day_10', 'Nikon_Z50','Light_disturbed', 'Batch_1_crop'): 0.3856,
    ( 'Day_10', 'Nikon_Z50','Light_disturbed', 'Batch_2_crop'): 0.2757,
    ( 'Day_10', 'Nikon_Z50','Light_disturbed', 'Batch_3_crop'): 0.3795,
    ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_1_crop'): 0.1551,
    ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_2_crop'): 0.1710,
    ( 'Day_12', 'Iphone_13_Pro_Max','Covered', 'Batch_3_crop'): 0.1539,
    ( 'Day_12', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_1_crop'): 0.1551,
    ( 'Day_12', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_2_crop'): 0.1710,
    ( 'Day_12', 'Iphone_13_Pro_Max','Light_disturbed', 'Batch_3_crop'): 0.1539,
    ( 'Day_12', 'Nikon_Z50','Covered', 'Batch_1_crop'): 0.1551,
    ( 'Day_12', 'Nikon_Z50','Covered', 'Batch_2_crop'): 0.1710,
    ( 'Day_12', 'Nikon_Z50','Covered', 'Batch_3_crop'): 0.1539,
    ( 'Day_12', 'Nikon_Z50','Light_disturbed', 'Batch_1_crop'): 0.1551,
    ( 'Day_12', 'Nikon_Z50','Light_disturbed', 'Batch_2_crop'): 0.1710,
    ( 'Day_12', 'Nikon_Z50','Light_disturbed', 'Batch_3_crop'): 0.1539,
}

# Load images and corresponding CPC concentrations
for day in days:
    for device in devices:
        for condition in conditions:
            for batch in batches:
                batch_dir = os.path.join(main_data_dir, day, device, condition, batch)
                cpc_concentration = cpc_concentration_map.get(batch, 0.0)  # Default to 0.0 if not found
                for img_file in os.listdir(batch_dir)[:160]:  # Limit to 160 images per batch
                    img_path = os.path.join(batch_dir, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error loading image: {img_path}")
                        continue
                    if img.shape[:2] != desired_pixel_size:
                        img = cv2.resize(img, desired_pixel_size)
                    img = img / 255.0
                    images.append(img)
                    cpc_concentrations.append(cpc_concentration)

X = np.array(images)
y = np.array(cpc_concentrations)

# Split the Data using train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Print the number of images for training and testing
print(f"Number of images for training: {len(X_train)}")
print(np.shape(X_train))
print(f"Number of images for testing: {len(X_test)}")
print(np.shape(X_test))

# # Build the CNN Model for Regression
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=desired_pixel_size + (3,)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='linear')  # Output layer with 3 neurons for low, medium, and high
])

# Set a smaller learning rate (e.g., 0.001)
custom_optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=custom_optimizer, loss='mean_squared_error', metrics=[RSquare(dtype=tf.float32, y_shape=(1,)), MeanAbsoluteError(), RootMeanSquaredError()])

# Train the Model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

print(history.history.keys())

# Advanced Visualizations
# Mean Squared Error vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.title('Training MSE vs Validation MSE')
plt.show()

# Mean Squared Error vs Epoch
plt.figure(figsize=(10, 5))
plt.plot(history.history['r_square'], label='Training R Square')
plt.plot(history.history['val_r_square'], label='Validation R Square')
plt.xlabel('Epochs')
plt.ylabel('R Square')
plt.legend()
plt.title('Training R Square vs Validation R Square')
plt.show()