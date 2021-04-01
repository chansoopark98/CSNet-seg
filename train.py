import tensorflow_datasets as tfds
import tensorflow as tf
import argparse
import time
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from model.model_builder import model_build
from preprocessing import pascal_prepare_dataset
import loss

parser = argparse.ArgumentParser()

parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=1)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=200)
parser.add_argument("--image_size",     type=int,   help="모델 입력 이미지 크기 설정", default=512)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.001)
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름", default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--backbone_model", type=str,   help="EfficientNet 모델 설정", default='B0')
parser.add_argument("--train_dataset",  type=str,   help="학습에 사용할 dataset 설정 coco or voc", default='voc')
parser.add_argument("--pretrain_mode",  type=bool,  help="저장되어 있는 가중치 로드", default=False)

args = parser.parse_args()
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
IMAGE_SIZE = [1024, 2048]
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
MODEL_NAME = args.backbone_model
TRAIN_MODE = args.train_dataset
CONTINUE_TRAINING = args.pretrain_mode

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


download_config = tfds.download.DownloadConfig(
                             manual_dir=DATASET_DIR+'/downloads', extract_dir=DATASET_DIR+'/cityscapes')

train_ds = tfds.load('cityscapes/semantic_segmentation', data_dir=DATASET_DIR, split='train', download_and_prepare_kwargs={"download_config":download_config})
valid_ds = tfds.load('cityscapes/semantic_segmentation', data_dir=DATASET_DIR, split='validation', download_and_prepare_kwargs={"download_config":download_config})
test_ds = tfds.load('cityscapes/semantic_segmentation', data_dir=DATASET_DIR, split='test', download_and_prepare_kwargs={"download_config":download_config})


train_data = train_ds.concatenate(valid_ds)




# number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
number_train = 3475
print("학습 데이터 개수", number_train)
# number_test = test_ds.reduce(0, lambda x, _: x + 1).numpy()
number_test = 1525
print("테스트 데이터 개수:", number_test)
optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)
# optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9)

training_dataset = pascal_prepare_dataset(train_data, BATCH_SIZE,
                                           train=True)
validation_dataset = pascal_prepare_dataset(test_ds, BATCH_SIZE,
                                            train=False)

print("백본 EfficientNet{0} .".format(MODEL_NAME))
model = model_build(MODEL_NAME, image_size=IMAGE_SIZE, pretrained=False)

if CONTINUE_TRAINING is True:
    model.load_weights(CHECKPOINT_DIR + '0217_main' + '.h5')

steps_per_epoch = number_train // BATCH_SIZE
validation_steps = number_test // BATCH_SIZE
print("학습 배치 개수:", steps_per_epoch)
print("검증 배치 개수:", validation_steps)
model.summary()


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, min_lr=1e-5, verbose=1)
checkpoint = ModelCheckpoint(CHECKPOINT_DIR + SAVE_MODEL_NAME + '.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)




model.compile(    optimizer=optimizer,
    loss=loss.weighted_cross_entropyloss)

    # metrics=[precision, recall, f1score])

history = model.fit(training_dataset,
                    validation_data=validation_dataset,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=EPOCHS,
                    callbacks=[reduce_lr, checkpoint, tensorboard])
