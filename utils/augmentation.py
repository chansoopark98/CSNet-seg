import tensorflow as tf

@tf.function
def get_indices_from_slice(top, left, height, width):
    a = tf.reshape(tf.range(top, top + height), [-1, 1])
    b = tf.range(left, left + width)
    A = tf.reshape(tf.tile(a, [1, width]), [-1])
    B = tf.tile(b, [height])
    indices = tf.stack([A, B], axis=1)

    return indices

@tf.function
def expand(image):
    image_shape = tf.cast(tf.shape(image), tf.float32)
    ratio = tf.random.uniform([], 1, 4)
    left = tf.math.round(tf.random.uniform([], 0, image_shape[1] * ratio - image_shape[1]))
    top = tf.math.round(tf.random.uniform([], 0, image_shape[0] * ratio - image_shape[0]))
    new_height = tf.math.round(image_shape[0] * ratio)
    new_width = tf.math.round(image_shape[1] * ratio)
    expand_image = tf.zeros((new_height, new_width, image_shape[2]), dtype=tf.float32)
    indices = get_indices_from_slice(int(top), int(left), int(image_shape[0]), int(image_shape[1]))
    expand_image = tf.tensor_scatter_nd_update(expand_image, indices, tf.reshape(image, [-1, 3]))
    expand_image = tf.image.resize(expand_image,(512, 1024))
    return expand_image

@tf.function
def random_lighting_noise(image):
    if tf.random.uniform([]) > 0.5:
        channels = tf.unstack(image, axis=-1)
        channels = tf.random.shuffle(channels)
        image = tf.stack([channels[0], channels[1], channels[2]], axis=-1)
    return image