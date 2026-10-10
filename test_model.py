import tensorflow as tf

model = tf.keras.models.load_model("currency_cnn.h5")

print("Model loaded successfully!")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)