import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential([
        
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), #input layer
        layers.MaxPooling2D((2, 2)),  #pooling layer 
        layers.Conv2D(64, (3, 3), activation='relu'), #hidden layer 1
        layers.MaxPooling2D((2, 2)), #pooling layer
        layers.Conv2D(64, (3, 3), activation='relu'), #hidden layer 2
        layers.Flatten(), #flatten 2d matrices to 1d vector 
        layers.Dense(64, activation='relu'), #fully connected dense layer
        layers.Dense(10, activation='softmax')  #output layer
    ])
    return model
