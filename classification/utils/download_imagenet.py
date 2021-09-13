from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds
import tensorflow as tf
import argparse
# import tensorflow_addons as tfa

AUTO = tf.data.experimental.AUTOTUNE


class CityScapes:
    def __init__(self, data_dir, image_size, batch_size, mode):
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


    def load_test(self, sample):
        img = sample['image_left']
        labels = sample['segmentation_label']

        # img = tf.image.resize(img, (512, 1024))
        # labels = tf.image.resize(labels, (512, 1024))

        img = tf.cast(img, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.int64)


        img = preprocess_input(img, mode='torch')


        return (img, labels)

    @tf.function
    def preprocess(self, sample):
        img = sample['image_left']
        labels = sample['segmentation_label']


        concat_img = tf.concat([img, labels], axis=-1)
        concat_img = tf.image.random_crop(concat_img, (self.image_size[0], self.image_size[1], 4))

        img = concat_img[:, :, :3]
        labels = concat_img[:, :, 3:]




        img = tf.cast(img, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.int64)

        img = preprocess_input(img, mode='torch')

        return (img, labels)

    @tf.function
    def preprocess_valid(self, sample):
        img = sample['image_left']
        labels = sample['segmentation_label']


        concat_img = tf.concat([img, labels], axis=-1)
        concat_img = tf.image.random_crop(concat_img, (self.image_size[0], self.image_size[1], 4))

        img = concat_img[:, :, :3]
        labels = concat_img[:, :, 3:]
        #
        # img = tf.image.resize(img, (512, 1024), method=tf.image.ResizeMethod.BILINEAR)
        # labels = tf.image.resize(labels, (512, 1024), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img = tf.cast(img, dtype=tf.float32)
        labels = tf.cast(labels, dtype=tf.int64)

        img = preprocess_input(img, mode='torch')

        return (img, labels)



    @tf.function
    def augmentation(self, img, labels):


        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_brightness(img, max_delta=0.4)
            # img = tf.image.random_brightness(img, max_delta=0.1)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_contrast(img, lower=0.7, upper=1.4)
            # img = tf.image.random_contrast(img, lower=0.1, upper=0.8)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_hue(img, max_delta=0.4)
        if tf.random.uniform([]) > 0.5:
            img = tf.image.random_saturation(img, lower=0.7, upper=1.4)
            # img = tf.image.random_saturation(img, lower=0.1, upper=0.8)
        # if tf.random.uniform([]) > 0.5:
        #     img = tfa.image.sharpness(img, factor=0.5)


        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            labels = tf.image.flip_left_right(labels)
        # random vertical flip
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_up_down(img)
            labels = tf.image.flip_up_down(labels)

        return (img, labels)

    def get_trainData(self, train_data):
        # num_parallel_calls=AUTO
        train_data = train_data.shuffle(buffer_size=1000)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
        train_data = train_data.prefetch(AUTO)
        train_data = train_data.repeat()
        train_data = train_data.padded_batch(self.batch_size)

        return train_data

    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.preprocess_valid, num_parallel_calls=AUTO)
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)
        return valid_data

    def get_testData(self, valid_data):
        valid_data = valid_data.map(self.load_test)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="배치 사이즈값 설정", default=256)
    parser.add_argument("--epoch", type=int, help="에폭 설정", default=100)
    parser.add_argument("--lr", type=float, help="Learning rate 설정", default=0.1)
    parser.add_argument("--weight_decay", type=float, help="Weight Decay 설정", default=0.00001)

    parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
    parser.add_argument("--checkpoint_dir", type=str, help="모델 저장 디렉토리 설정", default='./checkpoints/')
    parser.add_argument("--tensorboard_dir", type=str, help="텐서보드 저장 경로", default='tensorboard')
    parser.add_argument("--use_weightDecay", type=bool, help="weightDecay 사용 유무", default=False)
    parser.add_argument("--load_weight", type=bool, help="가중치 로드", default=False)
    parser.add_argument("--mixed_precision", type=bool, help="mixed_precision 사용", default=True)
    parser.add_argument("--distribution_mode", type=bool, help="분산 학습 모드 설정 mirror or multi", default='mirror')

    args = parser.parse_args()
    WEIGHT_DECAY = args.weight_decay
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epoch
    base_lr = args.lr

    DATASET_DIR = args.dataset_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    TENSORBOARD_DIR = args.tensorboard_dir
    IMAGE_SIZE = (224, 224)
    # IMAGE_SIZE = (None, None)
    USE_WEIGHT_DECAY = args.use_weightDecay
    LOAD_WEIGHT = args.load_weight
    MIXED_PRECISION = args.mixed_precision

    train_dataset_config = CityScapes(mode='train', data_dir=DATASET_DIR, image_size=IMAGE_SIZE,
                                            batch_size=BATCH_SIZE)
    valid_dataset_config = CityScapes(mode='validation', data_dir=DATASET_DIR, image_size=IMAGE_SIZE,
                                            batch_size=BATCH_SIZE)