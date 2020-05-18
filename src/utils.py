import io
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


class ObjectDict(dict):
    """
    Interface similar to an argparser
    """
    def __init__(self):
        pass

    def __setattr__(self, attr, value):
        self[attr] = value
        return self[attr]

    def __getattr__(self, attr):
        if attr.startswith('_'):
            # https://stackoverflow.com/questions/10364332/how-to-pickle-python-object-derived-from-dict
            raise AttributeError
        return dict(self)[attr]

    @property
    def __dict__(self):
        return dict(self)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    https://www.tensorflow.org/tensorboard/image_summaries
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
