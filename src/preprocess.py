import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

model = tf.keras.models.load_model('/Users/rammenon/Desktop/theMNISTmodel/models/cnn_mnist.keras')

def predict_digit(image_path):
    img = Image.open(image_path).convert('L') #mnist preprocessing criteria
    img = img.resize((28, 28))  
    img_array = np.array(img) / 255.0  
    img_array = img_array.reshape(1, 28, 28, 1)  
    predictions = model.predict(img_array)
    predicted_digit = np.argmax(predictions)

    return predicted_digit






image_path = 'image/guessnum.jpg'  #image taken from mnist itself due to incorrectly filtered self made input images. 
digit = predict_digit(image_path)
print(f'The predicted digit is: {digit}')


