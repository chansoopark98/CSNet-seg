import tensorflow as tf

@tf.function
def random_rescale_image_and_label(image, label, min_scale=0.5, max_scale=2.0):

  shape = tf.shape(image)
  height = tf.cast(shape[0], dtype=tf.float32)
  width = tf.cast(shape[1], dtype=tf.float32)
  scale = tf.random.uniform(
      [], minval=min_scale, maxval=max_scale, dtype=tf.float32)
  new_height = tf.cast(height * scale, dtype=tf.int32)
  new_width = tf.cast(width * scale, dtype=tf.int32)
  image = tf.image.resize(image, [new_height, new_width],
                                 method=tf.image.ResizeMethod.BILINEAR)

  label = tf.image.resize(label, [new_height, new_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return image, label


def random_crop_or_pad_image_and_label(image, label, crop_height=512, crop_width=1024):


  label = tf.cast(label, dtype=tf.float32)
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  image_and_label = tf.concat([image, label], axis=2)
  image_and_label_pad = tf.image.pad_to_bounding_box(
      image_and_label, 0, 0,
      tf.maximum(crop_height, image_height),
      tf.maximum(crop_width, image_width))
  image_and_label_crop = tf.image.random_crop(
      image_and_label_pad, [crop_height, crop_width, 4])

  image_crop = image_and_label_crop[:, :, :3]
  label_crop = image_and_label_crop[:, :, 3:]

  label_crop = tf.cast(label_crop, dtype=tf.int32)

  return image_crop, label_crop