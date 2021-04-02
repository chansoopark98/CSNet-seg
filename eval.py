import tensorflow_datasets as tfds
import tensorflow as tf
from model.model_builder import model_build
from preprocessing import pascal_prepare_dataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=512)
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/0402.h5')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
parser.add_argument("--calc_flops",  type=str,   help="모델 FLOPS 계산", default=False)
args = parser.parse_args()

BATCH_SIZE = 1
IMAGE_SIZE = [1024, 2048]
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
CALC_FLOPS = args.calc_flops

os.makedirs(DATASET_DIR, exist_ok=True)



download_config = tfds.download.DownloadConfig(
                             manual_dir=DATASET_DIR+'/downloads', extract_dir=DATASET_DIR+'/cityscapes')

test_data = tfds.load('cityscapes/semantic_segmentation', data_dir=DATASET_DIR, split='test', download_and_prepare_kwargs={"download_config":download_config})

number_test = test_data.reduce(0, lambda x, _: x + 1).numpy()
print("테스트 데이터 개수:", number_test)
NUM_CLASSES = 20
with open('./cityscape_labels') as f:
    CLASSES = f.read().splitlines()

# instantiate the datasets
validation_dataset = pascal_prepare_dataset(test_data, BATCH_SIZE,
                                            train=False)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = model_build(MODEL_NAME, pretrained=False)

print("모델 가중치 로드...")
model.load_weights(CHECKPOINT_DIR)

test_steps = number_test // BATCH_SIZE + 1
print("테스트 배치 개수 : ", test_steps)

import sklearn.metrics
conf_mat_total = np.zeros((NUM_CLASSES, NUM_CLASSES))
import matplotlib.pyplot as plt
for x, y in tqdm(validation_dataset, total=test_steps):
    pred = model.predict_on_batch(x)

    pred = tf.argmax(pred, axis=3) # (batch, 1024, 2048)
    pred = pred.numpy()
    pred = np.reshape(pred, (1024, 2048))
    y = np.reshape(y, (1024, 2048))
    pred = pred[y >= 0]
    y = y[y >=0 ]

    # test = tf.math.confusion_matrix(y_test, pred, num_classes=20)
    conf_mat = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES)))
    conf_mat_total += conf_mat



    ious = np.zeros((NUM_CLASSES,1))
    for l in range(NUM_CLASSES):
        ious[l] = conf_mat_total[l,l] / (np.sum(conf_mat_total[l,:]) +
                                         np.sum(conf_mat_total[:,l]) -
                                         conf_mat_total[l,l])

    print(ious)
    print('Mean IOU = {}\n'.format(np.mean(ious)))

