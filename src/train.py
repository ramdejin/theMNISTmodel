import tensorflow as tf
from dataset import load_data
from model import create_model

def train_model():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = create_model()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
    model.save('models/cnn_mnist.keras')
    return history
if __name__ == '__main__':
    train_model()
