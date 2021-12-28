import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import experimental as mixed_precision
# from ddrnet_23_slim.model.model_builder import seg_model_build
from model.model_builder import seg_model_build
import argparse
import time
import os
import tensorflow as tf
from tqdm import tqdm
from utils.load_datasets import CityScapes
from utils.cityscape_colormap import color_map
from utils.get_flops import get_flops

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
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

model = seg_model_build(image_size=IMAGE_SIZE, mode='seg', augment=True, weight_decay=WEIGHT_DECAY, num_classes=19)
weight_name = '_1227_best_miou'
# weight_name = '_1103_final_loss'
# weight_name = '_1030_best_loss'
model.load_weights(CHECKPOINT_DIR + weight_name + '.h5',by_name=True)
model.summary()
print(get_flops(model, batch_size=1))

class MeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.squeeze(y_true, axis=-1)
        y_pred += 1

        zeros_y_pred = tf.zeros(tf.shape(y_pred), tf.int64)
        zeros_y_pred += y_pred
        indices = tf.cast(tf.where(tf.equal(y_true, 0), 0, 1), tf.int64)

        y_true *= indices
        zeros_y_pred *= indices


        return super().update_state(y_true, zeros_y_pred, sample_weight)

metric = MeanIOU(20)
buffer = 0
batch_index = 1
save_path = './checkpoints/results/'+SAVE_MODEL_NAME+'/'
os.makedirs(save_path, exist_ok=True)


for x, y in tqdm(test_set, total=test_steps):
    pred = model.predict_on_batch(x)#pred = tf.nn.softmax(pred)
    #
    # arg_pred = tf.math.argmax(pred[1], axis=-1)
    # arg_pred = pred[1]
    # plt.imshow(arg_pred[0])
    # plt.show()

    #
    # arg_pred = tf.math.argmax(pred[2], axis=-1)
    # plt.imshow(arg_pred[0])
    # plt.show()
    #
    # arg_pred = tf.math.argmax(pred[3], axis=-1)
    # plt.imshow(arg_pred[0])
    # plt.show()

    pred = pred[0]
    arg_pred = tf.math.argmax(pred, axis=-1)

    for i in range(len(arg_pred)):
        metric.update_state(y[i], arg_pred[i])
        buffer = metric.result().numpy()
        print(buffer)
        # buffer += batch_miou

        r = arg_pred[i]
        g = arg_pred[i]
        b = arg_pred[i]

        for j in range(19):
            r = tf.where(tf.equal(r, j), color_map[j][0], r)
            g = tf.where(tf.equal(g, j), color_map[j][1], g)
            b = tf.where(tf.equal(b, j), color_map[j][2], b)

        r = tf.expand_dims(r, axis=-1)
        g = tf.expand_dims(g, axis=-1)
        b = tf.expand_dims(b, axis=-1)

        rgb_img = tf.concat([r, g, b], axis=-1)

        tf.keras.preprocessing.image.save_img(save_path+str(batch_index)+'.png', rgb_img)
        batch_index += 1




print("CityScapes validation 1024x2048 mIoU :  ", buffer)



