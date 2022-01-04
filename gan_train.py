from model.model_builder import build_dis, build_gen
import tensorflow as tf
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, mean_absolute_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow.keras.backend as K
from tqdm import tqdm
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow_io as tfio
from utils.cityscape_colormap import color_map, gt_color
from model.loss import Seg_loss
import psutil

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

def eacc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def l1(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def create_model_gen(input_shape, num_classes):
    model_input, model_output = build_gen(image_size=input_shape, num_classes=num_classes)

    model = tf.keras.Model(model_input, model_output)
    return model


def create_model_dis(input_shape):
    model_input, model_output = build_dis(image_size=input_shape)

    model = tf.keras.Model(model_input, model_output)
    return model


def create_model_gan(input_shape, generator, discriminator):
    input = Input(input_shape)

    gen_out = generator(input)
    # dis_out = discriminator(concatenate([gen_out, input], axis=3))
    gen_argmax = tf.math.argmax(gen_out, axis=-1)
    gen_argmax = tf.expand_dims(gen_argmax, axis=-1)
    gen_argmax = tf.cast(gen_argmax, tf.float32)
    half = 19. / 2.
    gen_argmax = (gen_argmax / half) - 1.
    dis_out = discriminator(gen_argmax)

    model = tf.keras.Model(inputs=[input], outputs=[dis_out, gen_out], name='dcgan')
    return model


def create_models(input_shape_gen, input_shape_dis, num_classes, lr, momentum, loss_weights):
    gen_loss = Seg_loss(aux_factor=1)
    gan_loss = Seg_loss(aux_factor=1)

    optimizer = Adam(learning_rate=lr, beta_1=momentum)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')  # tf2.4.1 이전

    model_gen = create_model_gen(input_shape=input_shape_gen, num_classes=num_classes)

    model_gen.compile(loss=gen_loss.ce_loss, optimizer=optimizer)

    model_dis = create_model_dis(input_shape=input_shape_dis)
    model_dis.trainable = False

    model_gan = create_model_gan(input_shape=input_shape_gen, generator=model_gen, discriminator=model_dis)
    model_gan.compile(
        loss=[binary_crossentropy, gan_loss.ce_loss],
        metrics=[eacc, 'accuracy'],
        loss_weights=loss_weights,
        optimizer=optimizer
    )

    model_gan.summary()

    model_dis.trainable = True
    model_dis.compile(loss=binary_crossentropy, optimizer=optimizer)

    return model_gen, model_dis, model_gan


@tf.function
def random_crop(img, labels, size=(512, 1024)):
    concat_img = tf.concat([img, labels], axis=-1)
    concat_img = tf.image.random_crop(concat_img, [size[0], size[1], 4])

    img = concat_img[:, :, :3]
    labels = concat_img[:, :, 3:]

    return img, labels

if __name__ == '__main__':
    EPOCHS = 200
    BATCH_SIZE = 4
    # LEARNING_RATE = 0.0005
    LEARNING_RATE = 0.0002
    MOMENTUM = 0.5
    LAMBDA1 = 1
    LAMBDA2 = 100
    INPUT_SHAPE_GEN = (512, 1024, 3)
    INPUT_SHAPE_DIS = (512, 1024, 1)
    NUM_CLASSES = 19
    DATASET_DIR = './datasets'
    WEIGHTS_GEN = './checkpoints/seg_GEN_'
    WEIGHTS_DIS = './checkpoints/seg_DIS_'
    WEIGHTS_GAN = './checkpoints/seg_GAN_'

    model_gen, model_dis, model_gan = create_models(
        input_shape_gen=INPUT_SHAPE_GEN,
        input_shape_dis=INPUT_SHAPE_DIS,
        num_classes=NUM_CLASSES,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        loss_weights=[LAMBDA1, LAMBDA2])

    # model_gen.load_weights(WEIGHTS_GEN + '0.h5')
    # model_dis.load_weights(WEIGHTS_DIS + '0.h5')
    # model_gan.load_weights(WEIGHTS_GAN + '0.h5')

    train_data = tfds.load('cityscapes/semantic_segmentation',
                           data_dir=DATASET_DIR, split='train')
    valid_data = tfds.load('cityscapes/semantic_segmentation',
                           data_dir=DATASET_DIR, split='validation[:10%]')

    number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()
    print("Train 데이터 개수", number_train)
    number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
    print("Validation 데이터 개수", number_valid)
    steps_per_epoch = number_train // BATCH_SIZE
    valid_per_epoch = number_valid // BATCH_SIZE

    train_data = train_data.shuffle(1024)
    train_data = train_data.padded_batch(BATCH_SIZE)
    # train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)

    valid_data = valid_data.padded_batch(BATCH_SIZE)
    # valid_data = valid_data.prefetch(tf.data.experimental.AUTOTUNE)

    demo_path = './demo_outputs/' + 'demo/'
    os.makedirs(demo_path, exist_ok=True)
    pbar = tqdm(train_data, total=steps_per_epoch, desc='Batch', leave=True, disable=False)

    for epoch in range(EPOCHS):

        batch_counter = 0
        toggle = True
        dis_res = 0
        index = 0
        for features in pbar:
            batch_counter += 1
            # ---------------------
            #  Train Discriminator
            # ---------------------
            img = features['image_left']
            labels = features['segmentation_label'] - 1

            shape = img.shape

            img = tf.image.resize(img, (INPUT_SHAPE_GEN[0], INPUT_SHAPE_GEN[1]),
                                  tf.image.ResizeMethod.BILINEAR)
            labels = tf.image.resize(labels, (INPUT_SHAPE_GEN[0], INPUT_SHAPE_GEN[1]),
                                     tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # concat_img = tf.concat([img, labels], axis=-1)
            # concat_img = tf.image.random_crop(concat_img, [BATCH_SIZE, INPUT_SHAPE_GEN[0], INPUT_SHAPE_GEN[1], 4])
            # img = concat_img[:, :, :, :3]
            # labels = concat_img[:, :, :, 3:]

            img = tf.cast(img, tf.float32)
            img = preprocess_input(img, mode='tf')

            # data augmentation
            if tf.random.uniform([], minval=0, maxval=1) > 0.5:
                img = tf.image.flip_left_right(img)


            if batch_counter % 2 == 0:
                toggle = not toggle
                if toggle:
                    x_dis = model_gen.predict(img)
                    x_dis = tf.math.argmax(x_dis, axis=-1)
                    x_dis = tf.cast(x_dis, tf.float32)
                    half = 19. / 2.
                    x_dis = (x_dis / half) - 1.
                    y_dis = tf.zeros((shape[0], 1))
                else:
                    # x_dis = tf.concat((l, ab), axis=3)
                    x_dis = labels
                    x_dis = tf.cast(x_dis, tf.float32)
                    half = 19. / 2.
                    x_dis = (x_dis / half) - 1.
                    y_dis = tf.random.uniform(shape=[shape[0]], minval=0.9, maxval=1)

                dis_res = model_dis.train_on_batch(x_dis, y_dis)

            model_dis.trainable = False
            x_gen = img
            y_gen = tf.ones((shape[0], 1))
            x_output = labels
            gan_res = model_gan.train_on_batch(x_gen, [y_gen, x_output])
            model_dis.trainable = True

            pbar.set_description(
                "Epoch : %d Dis loss: %f Gan total: %f Gan loss: %f CE loss: %f P_ACC: %f ACC: %f" % (epoch, dis_res,
                                                                                                     gan_res[0],
                                                                                                     gan_res[1],
                                                                                                     gan_res[2],
                                                                                                     gan_res[5],
                                                                                                     gan_res[6]))

        # if epoch % 5 == 0:
        model_gen.save_weights(WEIGHTS_GEN + str(epoch) + '.h5', overwrite=True)
        model_dis.save_weights(WEIGHTS_DIS + str(epoch) + '.h5', overwrite=True)
        model_gan.save_weights(WEIGHTS_GAN + str(epoch) + '.h5', overwrite=True)
        print( "Epoch : %d train end memory ==> %f" % (epoch, psutil.virtual_memory().used / 2 ** 30))
        metric = MeanIOU(20)

        os.makedirs(demo_path + str(epoch), exist_ok=True)
        # validation
        buffer = 0

        for valid_features in valid_data:

            original_input = valid_features['image_left']
            original_labels = valid_features['segmentation_label']

            shape = original_input.shape

            img = tf.image.resize(original_input, (INPUT_SHAPE_GEN[0], INPUT_SHAPE_GEN[1]),
                                  tf.image.ResizeMethod.BILINEAR)
            labels = tf.image.resize(original_labels, (INPUT_SHAPE_GEN[0], INPUT_SHAPE_GEN[1]),
                                     tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            img = tf.cast(img, dtype=tf.float32)
            labels = tf.cast(labels, dtype=tf.int64)
            img = preprocess_input(img, mode='tf')

            pred_gen = model_gen.predict(img)
            pred_gen = tf.math.argmax(pred_gen, axis=-1)

            metric.update_state(labels, pred_gen)
            buffer = metric.result().numpy()
            for i in range(len(pred_gen)):
                r = pred_gen[i]
                g = pred_gen[i]
                b = pred_gen[i]
                og_r = original_labels[i]
                og_g = original_labels[i]
                og_b = original_labels[i]

                for j in range(19):
                    r = tf.where(tf.equal(r, j), color_map[j][0], r)
                    g = tf.where(tf.equal(g, j), color_map[j][1], g)
                    b = tf.where(tf.equal(b, j), color_map[j][2], b)

                for j in range(20):
                    og_r = tf.where(tf.equal(og_r, j), gt_color[j][0], og_r)
                    og_g = tf.where(tf.equal(og_g, j), gt_color[j][1], og_g)
                    og_b = tf.where(tf.equal(og_b, j), gt_color[j][2], og_b)

                r = tf.expand_dims(r, axis=-1)
                g = tf.expand_dims(g, axis=-1)
                b = tf.expand_dims(b, axis=-1)


                rgb_img = tf.concat([r, g, b], axis=-1)
                og_gt = tf.concat([og_r, og_g, og_b], axis=-1)

                rows = 1
                cols = 3
                fig = plt.figure()

                ax0 = fig.add_subplot(rows, cols, 1)
                ax0.imshow(original_input[i])
                ax0.set_title('Input')
                ax0.axis("off")

                ax1 = fig.add_subplot(rows, cols, 2)
                ax1.imshow(rgb_img)
                ax1.set_title('Segmentation Results')
                ax1.axis("off")

                ax2 = fig.add_subplot(rows, cols, 3)
                ax2.imshow(og_gt)
                ax2.set_title('Label')
                ax2.axis("off")

                # plt.show()
                plt.savefig(demo_path + str(epoch) + '/' + str(index) + '.png', dpi=150)
                index += 1
        print( "Epoch : %d MIoU: %f" % (epoch, buffer))
        print( "Epoch : %d validation end memory ==> %f" % (epoch, psutil.virtual_memory().used / 2 ** 30))
