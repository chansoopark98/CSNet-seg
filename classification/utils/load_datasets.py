import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import sys
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
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_brightness(img, max_delta=0.4)
            # img = tf.image.random_brightness(img, max_delta=0.1)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_contrast(img, lower=0.7, upper=1.4)
            # img = tf.image.random_contrast(img, lower=0.1, upper=0.8)
        # if tf.random.uniform([]) > 0.5:
            img = tf.image.random_hue(img, max_delta=0.4)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_saturation(img, lower=0.7, upper=1.4)
            # img = tf.image.random_saturation(img, lower=0.1, upper=0.8)
        # if tf.random.uniform([]) > 0.5:
        #     img = tfa.image.sharpness(img, factor=0.5)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            labels = tf.image.flip_left_right(labels)
        # # random vertical flip
        # if tf.random.uniform([]) > 0.5:
        #     img = tf.image.flip_up_down(img)
        #     labels = tf.image.flip_up_down(labels)

        return (img, labels)


    @tf.function
    def preprocess(self, sample):
        img = tf.cast(sample['image'], dtype=tf.float32)
        label = tf.cast(sample['label'], dtype=tf.int64)
        # label = tf.one_hot(label[0], 20)
        # label = tf.reduce_max(tf.one_hot(label, 20, dtype=tf.int32), axis=0)
        # tf.print(label, output_stream=sys.stdout, summarize=-1)

        concat_img = tf.concat([img, label], axis=-1)
        concat_img = tf.image.random_crop(concat_img, (self.image_size[0], self.image_size[1], 4))

        img = concat_img[:, :, :3]
        label = concat_img[:, :, 3:]

        img = preprocess_input(img, mode='tf')

        return (img, label)

    def get_trainData(self, train_data):
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
        train_data = train_data.shuffle(buffer_size=10000)
        train_data = train_data.prefetch(AUTO)
        train_data = train_data.repeat()
        train_data = train_data.padded_batch(self.batch_size)

        return train_data

    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.preprocess, num_parallel_calls=AUTO)
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)
        return valid_data