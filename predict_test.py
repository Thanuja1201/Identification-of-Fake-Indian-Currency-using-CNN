
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained CNN model
model = tf.keras.models.load_model("currency_cnn.h5")

# Enter the path of one currency image
image_path = r"C:\Users\91903\.cache\kagglehub\datasets\jayaprakashpondy\indian-currency-dataset\versions\1\Indian Currency Dataset\test\real"

import os
image_name = os.listdir(image_path)[0]
full_path = os.path.join(image_path, image_name)

# Prepare the image
image = Image.open(full_path).convert("RGB")
image = image.resize((128, 128))
image_array = np.array(image, dtype=np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict
score = float(model.predict(image_array, verbose=0)[0][0])

print("Image tested:", image_name)
print("Model output score:", score)
print("Class mapping: fake = 0, real = 1")

if score >= 0.5:
    print("Prediction: REAL")
else:
    print("Prediction: FAKE")
