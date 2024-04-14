import tensorflow as tf
import numpy as np
import pandas as pd

# Define a function to parse the TFRecord file
def parse_tfrecord(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(example['label'], tf.int32)
    return image, label

# Load the TFRecord file and create a tf.data.Dataset
tfrecord_file = "C:/Users\Rajukumar\Downloads\Compressed\Plant Disease  Detection.v7i.tfrecord\train\Leaf-Disease.tfrecord"
dataset = tf.data.TFRecordDataset(tfrecord_file)
dataset = dataset.map(parse_tfrecord)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(dataset, epochs=10)