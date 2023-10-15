import tensorflow as tf
import cv2
import numpy as np

# Function to load the model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function to preprocess and predict an image
def predict_image(model, image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values

    # Make a prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    # Define class labels
    class_labels = ["Organic", "Recyclable"]

    # Return the predicted class label
    return class_labels[class_index]

if __name__ == "__main__":
    # Ask the user for the path to the model
    model_path = input("Enter the path to the saved model: ")

    # Load the model
    model = load_model(model_path)

    # Ask the user for the path to the image
    image_path = input("Enter the path to the image you want to classify: ")

    # Get the predicted class label
    predicted_label = predict_image(model, image_path)

    print(f"Predicted Class: {predicted_label}")
