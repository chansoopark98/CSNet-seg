from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds
from preprocessing import cityScapes
import tensorflow as tf

AUTO = tf.data.experimental.AUTOTUNE


class CityScapes:
    def __init__(self, data_dir, image_size, batch_size):
        """
        Args:
            data_dir: 데이터셋 상대 경로 ( default : './datasets/' )
            image_size: 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size

        self.train_data, self.valid_data, self.number_train, self.number_valid = self._load_datasets()

    def _load_datasets(self):
        train_data = tfds.load('cityscapes/semantic_segmentation',
                               data_dir=self.data_dir, split='train')
        valid_data = tfds.load('cityscapes/semantic_segmentation',
                               data_dir=self.data_dir, split='validation')

        number_train = self.train_data.reduce(0, lambda x, _: x + 1).numpy()
        print("학습 데이터 개수", self.number_train)
        number_valid = self.valid_data.reduce(0, lambda x, _: x + 1).numpy()
        print("검증 데이터 개수:", self.number_valid)

        return train_data, valid_data, number_train, number_valid

    def preprocess(self, sample):
        img = sample['image_left']
        labels = sample['segmentation_label']

        concat_img = tf.concat([img, labels], axis=-1)
        concat_img = tf.image.random_crop(concat_img, (512, 1024, 4))
        img = concat_img[:, :, :3]
        labels = concat_img[:, :, 3:]

        img = tf.cast(img, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.int64)

        img = preprocess_input(img, mode='tf')

        return (img, labels)\


    def augmentation(self, img, labels):
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

    def get_trainData(self):

        self.train_data = self.train_data.map(self.preprocess, num_parallel_calls=AUTO)
        self.train_data = self.train_data.shuffle(100).repeat()
        self.train_data = self.train_data.map(self.augmentation, num_parallel_calls=AUTO)
        self.train_data = self.train_data.batch(self.batch_size).prefetch(AUTO)
        return self.train_data

    def get_validData(self):

        self.valid_data = self.valid_data.map(self.preprocess)
        self.valid_data = self.valid_data.repeat()
        self.valid_data = self.valid_data.batch(self.batch_size).prefetch(AUTO)
        return self.valid_data
