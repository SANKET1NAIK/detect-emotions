import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score

# Define data directories
train_dir = "images\\train\\"
val_dir = "C:\\Users\\hp\\OneDrive\\Desktop\\face emotions\\images\\validation\\"
test_dir = "C:\\Users\\hp\\OneDrive\\Desktop\\face emotions\\images\\final test\\"

# Define data generators with image augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load data from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)


# Define CNN architecture
def build_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(48, 48, 1)
            ),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(
                7, activation="softmax"
            ),  # we are having 7 emotions thatwhy we use if we want more we have to change dense
        ]
    )
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


# Train the model, we are taking 10 epochs, if we want more good result we should do 100
model = build_model()
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=val_generator,
    validation_steps=len(val_generator),
)

# Evaluate model on validation set
val_loss, val_accuracy = model.evaluate(val_generator, steps=len(val_generator))
print("Validation Accuracy:", val_accuracy)

# Predict on test set
test_predictions = model.predict(test_generator, steps=len(test_generator))
test_labels = test_generator.classes
test_predictions = tf.argmax(test_predictions, axis=1)

# Calculate accuracy on test set
test_accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", test_accuracy)
