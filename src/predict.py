import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataset import load_data

def predict():
    (_, _), (x_test, y_test) = load_data()
    model = tf.keras.models.load_model('models/cnn_mnist.keras')
    index = np.random.randint(0, x_test.shape[0])
    image = x_test[index]
    label = y_test[index]
    prediction = model.predict(image[np.newaxis, ...])
    predicted_label = np.argmax(prediction)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Prediction: {predicted_label}, True Label: {np.argmax(label)}")
    plt.show()
if __name__ == '__main__':
    predict()
