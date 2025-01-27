import tensorflow as tf
from dataset import load_data

def evaluate_model():
    (_, _), (x_test, y_test) = load_data()
    model = tf.keras.models.load_model('models/cnn_mnist.h5')
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
if __name__ == '__main__':
    evaluate_model()
