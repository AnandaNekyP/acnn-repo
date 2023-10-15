import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, Attention
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC, Precision, Recall
import visualkeras
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

# Define paths
train_path = "../acnn-repo/DATASET/TRAIN"
test_path = "../acnn-repo/DATASET/TEST"

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,      # Rescale pixel values to [0, 1]
    rotation_range=40,      # Randomly rotate images
    width_shift_range=0.2,  # Randomly shift image width
    height_shift_range=0.2, # Randomly shift image height
    shear_range=0.2,        # Apply shear transformation
    zoom_range=0.2,         # Apply zoom transformation
    horizontal_flip=True,   # Randomly flip images horizontally
    fill_mode='nearest'     # Fill missing pixels with nearest values
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Batch size
batch_size = 32

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="rgb"
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="rgb"
)

# Define the model
model = Sequential()

input_shape = Input((224, 224, 3))

conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_shape)
maxpool1 = MaxPooling2D((2, 2))(conv1)
conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(maxpool1)
maxpool2 = MaxPooling2D((2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(maxpool2)
maxpool3 = MaxPooling2D((2, 2))(conv3)

AttentionLayer = Attention()([maxpool3, maxpool3])

flatten = Flatten()(AttentionLayer)

dense1 = Dense(128, activation='relu')(flatten)
dropout1 = Dropout(0.5)(dense1)
dense2 = Dense(64, activation='relu')(dropout1)
dropout2 = Dropout(0.5)(dense2)
output = Dense(train_generator.num_classes, activation='sigmoid')(dropout2)

model = tf.keras.Model(inputs=input_shape, outputs=output)

model.summary()

# Create a Visualkeras diagram for your model
visualkeras.layered_view(model, to_file='/content/model.png', legend=True, scale_xy=2)

# Compile the model with your custom optimizer and F1 score metric
model.compile(loss="binary_crossentropy",
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001),
              metrics=["accuracy", "AUC", "Precision", "Recall"])

# Train the model with more epochs
epochs = 20

# Train data
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    verbose=1
)

# Save the model architecture as JSON
model_json = model.to_json()
with open("model_acnn_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights as an HDF5 file
model.save_weights("model_acnn_weight.h5")

# Optionally, save the entire model including architecture and weights in one file
model.save("model_acnn_full.h5")
