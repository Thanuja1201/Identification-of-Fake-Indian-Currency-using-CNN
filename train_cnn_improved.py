import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

DATASET = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets",
    "jayaprakashpondy", "indian-currency-dataset",
    "versions", "1", "Indian Currency Dataset"
)

SIZE = (128, 128)
BATCH = 16
EPOCHS = 30

train_path = os.path.join(DATASET, "train")
test_path = os.path.join(DATASET, "test")

if not os.path.isdir(train_path) or not os.path.isdir(test_path):
    raise FileNotFoundError(
        "Dataset folders not found. Check the DATASET path."
    )

# Augmentation applies only to training images.
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=8,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    brightness_range=(0.9, 1.1),
    horizontal_flip=False
)

train = train_gen.flow_from_directory(
    train_path,
    target_size=SIZE,
    batch_size=BATCH,
    class_mode="binary",
    subset="training",
    seed=SEED,
    shuffle=True
)

validation_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

validation = validation_gen.flow_from_directory(
    train_path,
    target_size=SIZE,
    batch_size=BATCH,
    class_mode="binary",
    subset="validation",
    seed=SEED,
    shuffle=False
)

# Test images are never used for training or model selection.
test_gen = ImageDataGenerator(rescale=1.0 / 255)

test = test_gen.flow_from_directory(
    test_path,
    target_size=SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

print("Class labels:", train.class_indices)

# Balance the contribution of each training class.
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train.classes),
    y=train.classes
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        np.unique(train.classes), weights
    )
}

print("Class weights:", class_weights)

model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),

    layers.Conv2D(
        32, 3, padding="same", activation="relu"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        64, 3, padding="same", activation="relu"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.Conv2D(
        128, 3, padding="same", activation="relu"
    ),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),

    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "currency_cnn_improved_best.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]

model.fit(
    train,
    epochs=EPOCHS,
    validation_data=validation,
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluate the selected model on the untouched test set.
test.reset()
loss, accuracy = model.evaluate(test, verbose=1)

print("\nTest loss:", round(float(loss), 4))
print("Test accuracy: {:.2f}%".format(accuracy * 100))

test.reset()
probabilities = model.predict(test).ravel()
predictions = (probabilities >= 0.5).astype(int)

class_names = [
    name for name, index in sorted(
        test.class_indices.items(), key=lambda item: item[1]
    )
]

print("\nClass order:", test.class_indices)
print(classification_report(
    test.classes,
    predictions,
    labels=[0, 1],
    target_names=class_names,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(test.classes, predictions, labels=[0, 1]))

# Save under a new name; do not overwrite the original model.
model.save("currency_cnn_improved.h5")
print("\nSaved model: currency_cnn_improved.h5")