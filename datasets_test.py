import os

from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow_datasets as tfds
import tensorflow as tf
from utils.cityscape_colormap import color_map
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
        valid_data = tfds.load('cityscapes/semantic_segmentation',
                               data_dir=self.data_dir, split='validation')

        number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        # number_valid = 500
        print("검증 데이터 개수:", number_valid)

        return valid_data, number_valid

    def _load_train_datasets(self):
        train_data = tfds.load('cityscapes/semantic_segmentation',
                               data_dir=self.data_dir, split='train')


        number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
        # number_train = 2975
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
        labels = sample['segmentation_label']-1

        # gt = tf.cast(labels, tf.float32)
        # gt = tf.expand_dims(gt, axis=0)
        # grad_components = tf.image.sobel_edges(gt)
        #
        # grad_mag_components = grad_components ** 2
        #
        # grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)
        #
        # gt = tf.sqrt(grad_mag_square)
        #
        # mask = tf.cast(tf.where(gt != 0, 0.0, 1), tf.uint8)
        # labels *= mask

        y_true = labels
        # y_true += 1

        labels = tf.cast(y_true, tf.float32)
        labels = tf.expand_dims(labels, 0)
        grad_components = tf.image.sobel_edges(labels)

        grad_mag_components = grad_components ** 2

        grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)

        mask = tf.sqrt(grad_mag_square)

        # mask = tf.cast(mask, tf.uint8)
        #
        # y_true *= mask




        return (img, mask)

    @tf.function
    def preprocess_valid(self, sample):
        img = sample['image_left']
        y_true = sample['segmentation_label']


        # ### edge teest
        # orininal_label = y_true
        # edge_y_true = tf.cast(y_true, tf.float32)
        # edge_y_true = tf.expand_dims(edge_y_true, 0)
        # grad_components = tf.image.sobel_edges(edge_y_true)
        # grad_mag_components = grad_components ** 2
        # grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)
        #
        # mask = tf.sqrt(grad_mag_square)
        # ignore_mask = tf.where(y_true<0, 0, 1)
        #
        # y_true = tf.where(mask != 0, 1, 0)
        # ignore =y_true * ignore_mask


        #
        # edge_label = tf.cast(y_true, tf.float32)
        #
        # edge_label = tf.expand_dims(edge_label, 0)
        # grad_components = tf.image.sobel_edges(edge_label)
        #
        # grad_mag_components = grad_components ** 2
        #
        # grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)
        #
        # mask = tf.sqrt(grad_mag_square)
        #
        # y_true = tf.where(mask == 0, y_true, 0)

        return (img, y_true)



    @tf.function
    def augmentation(self, img, labels):

        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            labels = tf.image.flip_left_right(labels)


        return (img, labels)

    def get_trainData(self, train_data):
        # num_parallel_calls=AUTO
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        # train_data = train_data.shuffle(buffer_size=1000, reshuffle_each_iteration=True)
        # train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
        # train_data = train_data.prefetch(AUTO)
        # train_data = train_data.repeat()
        # train_data = train_data.padded_batch(self.batch_size)

        return train_data

    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.preprocess_valid, num_parallel_calls=AUTO)
        return valid_data

    def get_testData(self, valid_data):
        valid_data = valid_data.map(self.load_test)
        valid_data = valid_data.batch(self.batch_size).prefetch(AUTO)
        return valid_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, help="배치 사이즈값 설정", default=1)
    parser.add_argument("--epoch", type=int, help="에폭 설정", default=200)
    parser.add_argument("--lr", type=float, help="Learning rate 설정", default=0.01)
    parser.add_argument("--weight_decay", type=float, help="Weight Decay 설정", default=0.0005)

    parser.add_argument("--dataset_dir", type=str, help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
    parser.add_argument("--checkpoint_dir", type=str, help="모델 저장 디렉토리 설정", default='./checkpoints/')
    parser.add_argument("--tensorboard_dir", type=str, help="텐서보드 저장 경로", default='tensorboard')
    parser.add_argument("--use_weightDecay", type=bool, help="weightDecay 사용 유무", default=True)
    parser.add_argument("--load_weight", type=bool, help="가중치 로드", default=False)
    parser.add_argument("--mixed_precision", type=bool, help="mixed_precision 사용", default=True)
    parser.add_argument("--distribution_mode", type=bool, help="분산 학습 모드 설정 mirror or multi", default='mirror')

    args = parser.parse_args()
    WEIGHT_DECAY = args.weight_decay
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epoch
    base_lr = args.lr
    SAVE_MODEL_NAME = 'test'
    DATASET_DIR = args.dataset_dir
    CHECKPOINT_DIR = args.checkpoint_dir
    TENSORBOARD_DIR = args.tensorboard_dir
    IMAGE_SIZE = (512, 1024)
    # IMAGE_SIZE = (None, None)
    USE_WEIGHT_DECAY = args.use_weightDecay
    LOAD_WEIGHT = args.load_weight
    MIXED_PRECISION = args.mixed_precision
    DISTRIBUTION_MODE = args.distribution_mode

    train_dataset_config = CityScapes(DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, mode='validation')
    train_data = train_dataset_config.get_validData(train_dataset_config.valid_data)

    import matplotlib.pyplot as plt

    buffer = ''
    id_list = []
    stack = 0
    batch_index = 0
    img_path = './checkpoints/original/'
    label_path = './checkpoints/labels/'
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(label_path, exist_ok=True)

    for id in train_data.take(2975):
        x, y = id

        # img = tf.image.random_crop(x, (512, 1024, 3), seed=1000)
        # label = tf.image.random_crop(y, (512, 1024, 1), seed=1000)

        # plt.imshow(x)
        # plt.show()
        # plt.imshow(y)
        # plt.show()
        #
        # r = x
        # g = x
        # b = x
        #
        # for j in range(19):
        #     r = tf.where(tf.equal(r, j), color_map[j][0], r)
        #     g = tf.where(tf.equal(g, j), color_map[j][1], g)
        #     b = tf.where(tf.equal(b, j), color_map[j][2], b)
        #
        # # r = tf.expand_dims(r, axis=-1)
        # # g = tf.expand_dims(g, axis=-1)
        # # b = tf.expand_dims(b, axis=-1)
        #
        # rgb_img = tf.concat([r, g, b], axis=-1)
        #
        tf.keras.preprocessing.image.save_img(img_path + str(batch_index) + '.png', x)
        tf.keras.preprocessing.image.save_img(label_path + str(batch_index) + '.png', y)
        batch_index += 1

