from tensorflow.keras.applications.imagenet_utils import preprocess_input

import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE

@tf.function
def random_lighting_noise(image):
    if tf.random.uniform([]) > 0.5:
        channels = tf.unstack(image, axis=-1)
        channels = tf.random.shuffle(channels)
        image = tf.stack([channels[0], channels[1], channels[2]], axis=-1)
    return image

@tf.function
def data_augment(image):
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 랜덤 채도
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.15)  # 랜덤 밝기
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 랜덤 대비
    if tf.random.uniform([]) > 0.5:
        image = tf.image.random_hue(image, max_delta=0.2)  # 랜덤 휴 트랜스폼
    image = random_lighting_noise(image)

    return image


def prepare_input(sample, convert_to_normal=True):
    img = tf.cast(sample['image_left'], dtype=tf.float32)
    # img = img - image_mean 이미지 평균
    labels = tf.cast(sample['segmentation_label'], dtype=tf.float32)
    img = preprocess_input(img, mode='torch')

    return (img, labels)



def data_resize(sample):
    return tf.image.resize(sample, [512, 512])



def pascal_prepare_dataset(dataset, batch_size, train=False):
    classes = 34

    dataset = dataset.map(prepare_input, num_parallel_calls=AUTO)
    if train:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    # dataset = dataset.cache()
    dataset = dataset.map(data_resize, num_parallel_calls=AUTO)
    dataset = dataset.padded_batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset


# predict 할 때
def prepare_for_prediction(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img, [384, 384])
    img = preprocess_input(img, mode='torch')

    return img


def decode_img(img, image_size=[384, 384]):
    # 텐서 변환
    img = tf.image.decode_jpeg(img, channels=3)
    # 이미지 리사이징
    return tf.image.resize(img, image_size)

