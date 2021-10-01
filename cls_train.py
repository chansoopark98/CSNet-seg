from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from utils.callbacks import Scalar_LR
from classification.utils.load_datasets import GenerateDatasets
from classification.utils.callbacks import CustomLearningRateScheduler, lr_schedule
from classification.model.model_builder import seg_model_build
from classification.model.loss import Loss
import argparse
import time
import os
import tensorflow as tf
# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python cls_train.py
# LD_PRELOAD="/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4" python train.py

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",     type=int,   help="배치 사이즈값 설정", default=512)
parser.add_argument("--epoch",          type=int,   help="에폭 설정", default=100)
parser.add_argument("--lr",             type=float, help="Learning rate 설정", default=0.1)
parser.add_argument("--weight_decay",   type=float, help="Weight Decay 설정", default=0.0001)
parser.add_argument("--optimizer",     type=str,   help="Optimizer", default='sgd')
parser.add_argument("--model_name",     type=str,   help="저장될 모델 이름",
                    default=str(time.strftime('%m%d', time.localtime(time.time()))))
parser.add_argument("--dataset_dir",    type=str,   help="데이터셋 다운로드 디렉토리 설정", default='./datasets/')
parser.add_argument("--checkpoint_dir", type=str,   help="모델 저장 디렉토리 설정", default='./checkpoints/')
parser.add_argument("--tensorboard_dir",  type=str,   help="텐서보드 저장 경로", default='tensorboard')
parser.add_argument("--use_weightDecay",  type=bool,  help="weightDecay 사용 유무", default=False)
parser.add_argument("--load_weight",  type=bool,  help="가중치 로드", default=False)
parser.add_argument("--mixed_precision",  type=bool,  help="mixed_precision 사용", default=True)
parser.add_argument("--distribution_mode",  type=bool,  help="분산 학습 모드 설정 mirror or multi", default='mirror')

args = parser.parse_args()
WEIGHT_DECAY = args.weight_decay
OPTIMIZER_TYPE = args.optimizer
BATCH_SIZE = args.batch_size
EPOCHS = args.epoch
base_lr = args.lr
SAVE_MODEL_NAME = args.model_name
DATASET_DIR = args.dataset_dir
CHECKPOINT_DIR = args.checkpoint_dir
TENSORBOARD_DIR = args.tensorboard_dir
IMAGE_SIZE = (224, 224)
# IMAGE_SIZE = (None, None)
USE_WEIGHT_DECAY = args.use_weightDecay
LOAD_WEIGHT = args.load_weight
MIXED_PRECISION = args.mixed_precision
DISTRIBUTION_MODE = args.distribution_mode

if MIXED_PRECISION:
    policy = mixed_precision.Policy('mixed_float16', loss_scale=1024)
    mixed_precision.set_policy(policy)

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


if DISTRIBUTION_MODE == 'multi':
    mirrored_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)

else:
    # mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    mirrored_strategy = tf.distribute.MirroredStrategy()


with mirrored_strategy.scope():
    print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

    train_dataset_config = GenerateDatasets(mode='train', data_dir=DATASET_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)
    valid_dataset_config = GenerateDatasets(mode='validation', data_dir=DATASET_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

    train_data = train_dataset_config.get_trainData(train_dataset_config.train_data)
    train_data = mirrored_strategy.experimental_distribute_dataset(train_data)
    valid_data = valid_dataset_config.get_validData(valid_dataset_config.valid_data)
    valid_data = mirrored_strategy.experimental_distribute_dataset(valid_data)

    steps_per_epoch = train_dataset_config.number_train // BATCH_SIZE

    validation_steps = valid_dataset_config.number_valid // BATCH_SIZE
    print("학습 배치 개수:", steps_per_epoch)
    print("검증 배치 개수:", validation_steps)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, min_lr=1e-5, verbose=1)

    checkpoint_val_loss = ModelCheckpoint(CHECKPOINT_DIR + '_' + SAVE_MODEL_NAME + '_imagenet_best_loss.h5',
                                          monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1)

    checkpoint_val_accuracy = ModelCheckpoint(CHECKPOINT_DIR + '_' + SAVE_MODEL_NAME + '_imagenet_best_top1_accuracy.h5',
                                          monitor='val_top1', save_best_only=True, save_weights_only=True, verbose=1, mode='max')


    testCallBack = Scalar_LR('test', TENSORBOARD_DIR)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, write_graph=True, write_images=True)

    if OPTIMIZER_TYPE == 'sgd':
        optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=base_lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=base_lr)

    if MIXED_PRECISION:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전


    callback = [tensorboard, testCallBack, CustomLearningRateScheduler(lr_schedule),
                checkpoint_val_loss, checkpoint_val_accuracy]

    loss = Loss(BATCH_SIZE)

    model = seg_model_build(image_size=IMAGE_SIZE, mode='cls', augment=False, weight_decay=WEIGHT_DECAY,
                            optimizer=OPTIMIZER_TYPE)
    model.compile(
        optimizer=optimizer,
        loss=loss.sparse_categorical_loss,
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top1'),
                 tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')])

    if LOAD_WEIGHT:
        weight_name = '_0916_best_backbone_accuracy'
        model.load_weights(CHECKPOINT_DIR + weight_name + '.h5')

    model.summary()

    history = model.fit(train_data,
            validation_data=valid_data,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=EPOCHS,
            callbacks=callback)


