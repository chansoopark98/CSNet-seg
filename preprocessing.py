from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow as tf
AUTO = tf.data.experimental.AUTOTUNE

def prepare_cityScapes(sample):
    # img = sample['image_left']
    # labels = sample['segmentation_label']

    img = sample['image_left']
    labels = sample['segmentation_label']

    concat_img = tf.concat([img, labels], axis=-1)
    concat_img = tf.image.random_crop(concat_img, (512, 1024, 4))
    img = concat_img[:, :, :3]
    labels = concat_img[:, :, 3:]



    img = tf.cast(img, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.int64)

    img = preprocess_input(img, mode='tf')

    return (img, labels)


@tf.function
def data_augment_cityScapes(img, labels):
    # concat_img = tf.concat([img, labels], axis=-1)
    # concat_img = tf.image.random_crop(concat_img, (512, 1024, 4))
    # img = concat_img[:, :, :3]
    # labels = concat_img[:, :, 3:]
    #
    # img = tf.image.resize(img, (512, 1024))
    # labels = tf.image.resize(labels, (512, 1024))

    # img /= 255
    # image_mean = (0.28689554, 0.32513303, 0.28389177)
    # image_std = (0.18696375, 0.19017339, 0.18720214)
    # img = (img - image_mean) / image_std # Cityscapes mean, std 실험
    # img = tf.cast(img, dtype=tf.float32)
    # labels = tf.cast(labels, dtype=tf.int64)

    # img = preprocess_input(img, mode='tf')

    if tf.random.uniform([]) > 0.5:
        img = tf.image.random_brightness(img, max_delta=0.4)
    if tf.random.uniform([]) > 0.5:
        img = tf.image.random_contrast(img, lower=0.7, upper=1.4)
    if tf.random.uniform([]) > 0.5:
        img = tf.image.random_hue(img, max_delta=0.4)
    if tf.random.uniform([]) > 0.5:
        img = tf.image.random_saturation(img, lower=0.7, upper=1.4)
    if tf.random.uniform([]) > 0.5:
        img = tf.image.flip_left_right(img)
        labels = tf.image.flip_left_right(labels)


    return (img, labels)


def cityScapes_val_resize(img, labels):
    # img = tf.image.resize(img, (512, 1024))
    # labels = tf.image.resize(labels, (512, 1024))
    concat_img = tf.concat([img, labels], axis=-1)
    concat_img = tf.image.random_crop(concat_img, (512, 1024, 4))
    img = concat_img[:, :, :3]
    labels = concat_img[:, :, 3:]

    img /= 255
    image_mean = (0.28689554, 0.32513303, 0.28389177)
    image_std = (0.18696375, 0.19017339, 0.18720214)
    img = (img - image_mean) / image_std # Cityscapes mean, std 실험
    img = tf.cast(img, dtype=tf.float32)

    # img = preprocess_input(img, mode='tf')
    # img /= 255
    # img = tf.cast(img, dtype=tf.float32)
    labels = tf.cast(labels, dtype=tf.int64)
    return (img, labels)

def cityScapes(dataset, image_size=None, batch_size=None, train=False):

    if train:
        dataset = dataset.map(prepare_cityScapes, num_parallel_calls=AUTO)
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()
        dataset = dataset.map(data_augment_cityScapes, num_parallel_calls=AUTO)
        # dataset = dataset.map(cityScapes_resize, num_parallel_calls=AUTO)

    else:
        dataset = dataset.map(prepare_cityScapes)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset

