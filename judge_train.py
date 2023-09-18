"""
Code to train the six pixel model in order to replicate "AI safety via debate" 
by Geoffrey Irving, Paul Christiano, and Dario Amodei. For more information,
the original paper can be found at:
    https://arxiv.org/abs/1805.00899

Much of the code in this document, expect for model architecture was taken
from:
    https://keras.io/examples/vision/mnist_convnet/
(Licensed under Apache License Version 2.0)
Details: The original code was modified to accommodate the different input shape,
training details, and model architecture. Additional data processing of the
MNIST dataset was also added.

The architecture for this model was taken from the TensorFlow documentation:
    https://www.tensorflow.org/tutorials/quickstart/advanced
(Licensed under Apache License Version 2.0)
"""

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import numpy as np
import random
import argparse

# Model / data parameters
batch_size = 128
num_classes = 10
input_shape = (28, 28, 2)  # NOTE: changed this to support mask channel
steps_per_epoch = 2
epochs = 1


def pick_random_non_zero_pixels(img, num_pixels):
    """
    Picks random non-zero pixel locations from the given image.

    Returns a list of `num_pixels` length containing the locations of non-zero
    pixels from the image.

    Args:
        img (numpy.ndarray): The input image with shape (height, width, channels).
        num_pixels (int): Number of non-zero pixel locations to select.

    Returns:
        list[tuple[int, int]]: List of tuples where each tuple is the
        (row, column) location of a non-zero pixel.

    Note:
        Assumes the input image has one channel.
    """
    non_zero_locs = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j][0] != 0:
                non_zero_locs.append((i, j))
    rv = []
    for _ in range(num_pixels):
        rv.append(random.choice(non_zero_locs))
    return rv


def apply_mask_to_img(img, num_pixels_unmasked):
    """
    Applies a mask to the given image.

    The number of pixels that retain thier values is determined by
    `num_pixels_unmasked`. All other pixel values are set to 0. Each pixel in the
    returned array is length 2:
        [(1 if pixel is revealed; else 0), (value of pixel if revealed)]

    Args:
        img (numpy.ndarray): The input image with shape (height, width, channels).
        num_pixels_unmasked (int): Number of pixels to remain unmasked.

    Returns:
        numpy.ndarray: A masked version of the input image (shape = (28, 28, 2)).

    Note:
        Assumes the input image has a shape of (28, 28, 1).
    """
    unmasked_locs = pick_random_non_zero_pixels(img, num_pixels_unmasked)
    rv = np.zeros((28, 28, 2))
    for loc in unmasked_locs:
        rv[loc[0]][loc[1]][0] = 1
        rv[loc[0]][loc[1]][1] = img[loc[0]][loc[1]][0]
    return rv


# APPLYS MASKS TO TRAINING IMAGES
def generate_training(x_train, y_train, num_pixels_unmasked):
    while True:
        x_train_transformed = np.empty((128, 28, 28, 2))
        start = random.randint(0, 60000 - 128)

        for i in range(128):
            x_train_transformed[i] = apply_mask_to_img(
                x_train[start + i], num_pixels_unmasked
            )
        yield x_train_transformed, y_train[start : start + 128]


def declare_model_and_training_data_generator(x_train, y_train, num_pixels_unmasked):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: generate_training(x_train, y_train, num_pixels_unmasked),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, 28, 28, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 10), dtype=tf.float32),
        ),
    )
    return model, train_dataset


def main():
    parser = argparse.ArgumentParser(
        description="The number of pixels revealed in each image durring training and testing."
    )
    parser.add_argument(
        "--num_pixels",
        choices=["4", "6"],
        required=True,
        help='Pass in the argument "4" or "6" to choose how many pixels will be revealed during training and testing',
    )

    args = parser.parse_args()
    num_pixels_unmasked = int(args.num_pixels)

    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # APPLYS MASKS TO TESTING IMAGES
    x_test_transformed = np.empty((len(x_test), 28, 28, 2))

    for i in range(len(x_test)):
        x_test_transformed[i] = apply_mask_to_img(x_test[i], num_pixels_unmasked)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model, train_dataset = declare_model_and_training_data_generator(
        x_train, y_train, num_pixels_unmasked
    )

    model.fit(
        train_dataset,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )

    score = model.evaluate(x_test_transformed, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    paths_for_saved_models = [".h5", ".keras"]
    for i in range(len(paths_for_saved_models)):
        if num_pixels_unmasked == 4:
            paths_for_saved_models[i] = "four_pixel_judge" + paths_for_saved_models[i]
        else:
            paths_for_saved_models[i] = "six_pixel_judge" + paths_for_saved_models[i]

    paths_for_saved_models[1] = "models_keras_format/" + paths_for_saved_models[1]

    for path in paths_for_saved_models:
        model.save_weights(path)


if __name__ == "__main__":
    main()
