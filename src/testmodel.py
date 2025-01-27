from tensorflow.keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

model = tf.keras.models.load_model('/Users/rammenon/Desktop/theMNISTmodel/models/cnn_mnist.keras') #load saved model from native directory
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0  
x_test = x_test.reshape(-1, 28, 28, 1)  
y_test = to_categorical(y_test, num_classes=10)  
print("Model Test Accuracy:", model.evaluate(x_test, y_test))

