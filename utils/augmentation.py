import tensorflow as tf

@tf.function
def random_lighting_noise(image):
    if tf.random.uniform([]) > 0.5:
        channels = tf.unstack(image, axis=-1)
        channels = tf.random.shuffle(channels)
        image = tf.stack([channels[0], channels[1], channels[2]], axis=-1)
    return image