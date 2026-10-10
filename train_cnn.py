import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

DATASET = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets",
    "jayaprakashpondy", "indian-currency-dataset",
    "versions", "1", "Indian Currency Dataset"
)

SIZE = (128, 128)
BATCH = 16
EPOCHS = 15

train_path = os.path.join(DATASET, "train")
test_path = os.path.join(DATASET, "test")

# Split the training folder into training and validation subsets.
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=8,
    zoom_range=0.1
)

train = train_gen.flow_from_directory(
    train_path,
    target_size=SIZE,
    batch_size=BATCH,
    class_mode="binary",
    subset="training",
    seed=42
)

validation = train_gen.flow_from_directory(
    train_path,
    target_size=SIZE,
    batch_size=BATCH,
    class_mode="binary",
    subset="validation",
    seed=42
)

# Keep the test set separate from training.
test_gen = ImageDataGenerator(rescale=1.0 / 255)

test = test_gen.flow_from_directory(
    test_path,
    target_size=SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

print("Class labels:", train.class_indices)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train,
    epochs=EPOCHS,
    validation_data=validation,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]
)

# Final evaluation on the separate test set.
test.reset()
loss, accuracy = model.evaluate(test, verbose=1)
print(f"\nTest loss: {loss:.4f}")
print(f"Test accuracy: {accuracy * 100:.2f}%")

test.reset()
probabilities = model.predict(test).ravel()
predictions = (probabilities >= 0.5).astype(int)

print("\nClass order:", test.class_indices)
print(classification_report(
    test.classes,
    predictions,
    target_names=list(test.class_indices.keys()),
    zero_division=0
))
print("Confusion matrix:")
print(confusion_matrix(test.classes, predictions))

model.save("currency_cnn.h5")
print("\nSaved trained model: currency_cnn.h5")
