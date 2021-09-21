# Sunflower or Daisy
#
# Image

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

dataset_url = "data/sunflowers_and_daisies/"
daisy_data_dir = tf.keras.utils.get_file('daisy', origin=dataset_url, untar=True)
daisy_data_dir = pathlib.Path(daisy_data_dir)

daisy_image_count = len(list(daisy_data_dir.glob('*/*.jpg')))
print(daisy_image_count)