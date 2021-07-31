from tensorflow.keras.mixed_precision import experimental as mixed_precision
from model.model_builder import seg_model_build
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
from utils.load_datasets import CityScapes
import tensorflow_datasets as tfds

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=8)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=100)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0005)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=False)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=True)
parser.add_argument("--distribution_mode",  type=bool,  help="분산 학습 모드 설정 mirror or multi", default='mirror')

args = parser.parse_args()
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
IMAGE_SIZE = (1024, 2048)
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode

if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Create Dataset
# dataset_config = CityScapes(DATASET_DIR, IMAGE_SIZE, BATCH_SIZE)
dataset = CityScapes(DATASET_DIR, IMAGE_SIZE, BATCH_SIZE, 'valid')

test_steps = dataset.number_valid // BATCH_SIZE

test_set = dataset.get_testData(dataset.valid_data)


# if DISTRIBUTION_MODE == 'multi':
#     mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
#         tf.distribute.experimental.CollectiveCommunication.NCCL)
#
# else:
#     mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
# print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
#
# with mirrored_strategy.scope():

model = seg_model_build(image_size=IMAGE_SIZE)

weight_name = '_0731_best_miou'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

model.summary()


import matplotlib.pyplot as plt

class MeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.squeeze(y_true, -1)
        # y_pred = tf.argmax(y_pred, axis=-1)

        # mask = tf.where(y_true>0, True, False)
        # y_true = tf.boolean_mask(y_true, mask)
        # y_pred = tf.expand_dims(y_pred, axis=-1)
        # y_pred = tf.boolean_mask(y_pred, mask)

        return super().update_state(y_true, y_pred, sample_weight)

metric = MeanIOU(20)
buffer = 0
for x, y in tqdm(test_set, total=test_steps):
    pred = model.predict_on_batch(x)#pred = tf.nn.softmax(pred)
    pred = tf.argmax(pred, axis=-1)
    for i in range(len(pred)):
        metric.update_state(y[i], pred[i])
        buffer += metric.result().numpy()

        plt.imshow(pred[i])
        plt.show()


print("CityScapes validation 1024x2048 mIoU :  ", buffer/dataset.number_valid)






