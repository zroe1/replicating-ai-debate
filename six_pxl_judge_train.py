from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import numpy as np
import random

# NOTE: Change the following list to modify the locations and formats of the
# saved models
PATHS_FOR_SAVED_MODELS = ['model_weights.h5', 'model_weights.keras']

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
      if (img[i][j][0] != 0):
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
def generate_training():
  while True:
    x_train_transformed = np.empty((128, 28, 28, 2))
    start = random.randint(0, 60000-128)

    for i in range(128):
      x_train_transformed[i] = apply_mask_to_img(x_train[start+i], 6)
    yield x_train_transformed, y_train[start:start+128]

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 2) # NOTE: changed this to support mask channel

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("x_train shape:", x_test.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# APPLYS MASKS TO TESTING IMAGES
x_test_transformed = np.empty((len(x_test), 28, 28, 2))

for i in range(len(x_test)):
  x_test_transformed[i] = apply_mask_to_img(x_test[i], 6)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # layers.Dense(10),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()

batch_size = 128
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

train_dataset = tf.data.Dataset.from_generator(
    generate_training,
    output_signature=(
        tf.TensorSpec(shape=(batch_size, 28, 28, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(batch_size, 10), dtype=tf.float32)
    )
)

model.fit(train_dataset, batch_size=batch_size, steps_per_epoch=30000, epochs=1)

score = model.evaluate(x_test_transformed, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


for path in PATHS_FOR_SAVED_MODELS:
    model.save_weights(path)
