import os
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

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
EPOCHS = 15

train_path = os.path.join(DATASET, "train")
test_path = os.path.join(DATASET, "test")

train_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2,
    rotation_range=5,
    zoom_range=0.08
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
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
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

test_gen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

test = test_gen.flow_from_directory(
    test_path,
    target_size=SIZE,
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

print("Class labels:", train.class_indices)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train.classes),
    y=train.classes
)

class_weights = {
    int(label): float(weight)
    for label, weight in zip(np.unique(train.classes), weights)
}

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(128, 128, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(128, 128, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "currency_mobilenet_best.h5",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
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

test.reset()
loss, accuracy = model.evaluate(test, verbose=1)

print("\nTest loss:", round(float(loss), 4))
print("Test accuracy: {:.2f}%".format(accuracy * 100))

test.reset()
probabilities = model.predict(test).ravel()
predictions = (probabilities >= 0.5).astype(int)

class_names = [
    name for name, index in sorted(
        test.class_indices.items(),
        key=lambda item: item[1]
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

model.save("currency_mobilenet.h5")
print("\nSaved model: currency_mobilenet.h5")
