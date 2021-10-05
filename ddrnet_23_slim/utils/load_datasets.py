import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import numpy as np
AUTO = tf.data.experimental.AUTOTUNE

class GenerateDatasets:
    def __init__(self, mode, data_dir, image_size, batch_size):
        """
        Args:
            data_dir: 데이터셋 상대 경로 ( default : './datasets/' )
            image_size: 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size

        if mode == 'train':
            self.train_data, self.number_train = self._load_train_datasets()
        else:
            self.valid_data, self.number_valid = self._load_valid_datasets()


    def _load_valid_datasets(self):
        valid_data = tfds.load('imagenet2012',
                               data_dir=self.data_dir, split='validation')

        # number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        number_valid = 50000
        print("검증 데이터 개수:", number_valid)

        return valid_data, number_valid

    def _load_train_datasets(self):
        train_data = tfds.load('imagenet2012',
                               data_dir=self.data_dir, split='train')


        # number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        number_train = 1281167
        print("학습 데이터 개수", number_train)

        return train_data, number_train


    @tf.function
    def augmentation(self, img, labels):
        img = tf.cast(img, tf.float32)
        if tf.random.uniform([], 0, 1) > 0.75:
            img = tf.image.random_hue(img, 0.08)
            img = tf.image.random_saturation(img, 0.6, 1.6)
            img = tf.image.random_brightness(img, 0.05)
            img = tf.image.random_contrast(img, 0.7, 1.3)

        if tf.random.uniform([], 0, 1) > 0.5:
            img = tf.image.flip_left_right(img)

        if tf.random.uniform([], 0, 1) > 0.5:
            img = self.zoom(img)

        if tf.random.uniform([], 0, 1) > 0.5:
            img = self.rotate(img)

        img = preprocess_input(img, mode='torch')

        return (img, labels)

    @tf.function
    def zoom(self, x, scale_min=0.8, scale_max=1.2):
        h, w, _ = x.shape
        scale = tf.random.uniform([], scale_min, scale_max)
        nh = h * scale
        nw = w * scale
        x = tf.image.resize(x, (nh, nw))
        x = tf.image.resize_with_crop_or_pad(x, h, w)
        return x

    @tf.function
    def rotate(self, x, angle=(-40, 40)):
        angle = tf.random.uniform([], angle[0], angle[1], tf.float32)
        theta = np.pi * angle / 180

        x = tfa.image.rotate(x, theta)
        return x

    @tf.function
    def resize_image(self, image, shape=(224, 224)):
        target_width = shape[0]
        target_height = shape[1]
        initial_width = tf.shape(image)[0]
        initial_height = tf.shape(image)[1]
        im = image
        ratio = 0
        if (initial_width < initial_height):
            ratio = tf.cast(256 / initial_width, tf.float32)
            h = tf.cast(initial_height, tf.float32) * ratio
            im = tf.image.resize(im, (256, h), method="bicubic")
        else:
            ratio = tf.cast(256 / initial_height, tf.float32)
            w = tf.cast(initial_width, tf.float32) * ratio
            im = tf.image.resize(im, (w, 256), method="bicubic")
        width = tf.shape(im)[0]
        height = tf.shape(im)[1]
        startx = width // 2 - (target_width // 2)
        starty = height // 2 - (target_height // 2)
        im = tf.image.crop_to_bounding_box(im, startx, starty, target_width, target_height)
        return im

    @tf.function
    def preprocess(self, sample):
        img = sample['image']
        label = sample['label']

        # img = tf.image.random_crop(img, (self.image_size[0], self.image_size[1], 3))
        # img = tf.image.resize(img, (self.image_size[0], self.image_size[1]))
        # img = tf.image.resize_with_pad(img, 256, 256)
        img = tf.image.resize_with_crop_or_pad(img, 224, 224)

        # img = tf.image.random_crop(img, (self.image_size[0], self.image_size[1], 3))

        # img = self.resize_image(img, (224, 224))

        return (img, label)

    @tf.function
    def preprocess_valid(self, sample):
        img = tf.cast(sample['image'], tf.float32)
        label = sample['label']

        # img = tf.image.random_crop(img, (self.image_size[0], self.image_size[1], 3))
        # img = tf.image.resize(img, (self.image_size[0], self.image_size[1]))
        img = self.resize_image(img, (224, 224))

        img = preprocess_input(img, mode='torch')

        return (img, label)

    def get_trainData(self, train_data):
        train_data = train_data.shuffle(buffer_size=10000)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
        train_data = train_data.prefetch(AUTO)
        train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.repeat()

        return train_data

    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.preprocess_valid, num_parallel_calls=AUTO)
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)
        return valid_data