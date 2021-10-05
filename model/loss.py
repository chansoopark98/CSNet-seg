import tensorflow as tf
import itertools
from typing import Any, Optional
_EPSILON = tf.keras.backend.epsilon()

def sparse_categorical_focal_loss(y_true, y_pred, gamma, *,
                                  class_weight: Optional[Any] = None,
                                  from_logits: bool = False, axis: int = -1
                                  ) -> tf.Tensor:

    # Process focusing parameter
    gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
    gamma_rank = gamma.shape.rank
    scalar_gamma = gamma_rank == 0

    # Process class weight
    if class_weight is not None:
        class_weight = tf.convert_to_tensor(class_weight,
                                            dtype=tf.dtypes.float32)

    # Process prediction tensor
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred_rank = y_pred.shape.rank
    if y_pred_rank is not None:
        axis %= y_pred_rank
        if axis != y_pred_rank - 1:
            # Put channel axis last for sparse_softmax_cross_entropy_with_logits
            perm = list(itertools.chain(range(axis),
                                        range(axis + 1, y_pred_rank), [axis]))
            y_pred = tf.transpose(y_pred, perm=perm)
    elif axis != -1:
        raise ValueError(
            f'Cannot compute sparse categorical focal loss with axis={axis} on '
            'a prediction tensor with statically unknown rank.')
    y_pred_shape = tf.shape(y_pred)

    # Process ground truth tensor
    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    y_true_rank = y_true.shape.rank

    if y_true_rank is None:
        raise NotImplementedError('Sparse categorical focal loss not supported '
                                  'for target/label tensors of unknown rank')

    reshape_needed = (y_true_rank is not None and y_pred_rank is not None and
                      y_pred_rank != y_true_rank + 1)
    if reshape_needed:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

    if from_logits:
        logits = y_pred
        probs = tf.nn.softmax(y_pred, axis=-1)
    else:
        probs = y_pred
        logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=logits,
    )

    y_true_rank = y_true.shape.rank
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)
    if not scalar_gamma:
        gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)
    focal_modulation = (1 - probs) ** gamma
    loss = focal_modulation * xent_loss

    if class_weight is not None:
        class_weight = tf.gather(class_weight, y_true, axis=0,
                                 batch_dims=y_true_rank)
        loss *= class_weight

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    return loss


@tf.keras.utils.register_keras_serializable()
class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma, class_weight: Optional[Any] = None,
                 from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.class_weight = class_weight
        self.from_logits = from_logits

    def get_config(self):
        config = super().get_config()
        config.update(gamma=self.gamma, class_weight=self.class_weight,
                      from_logits=self.from_logits)
        return config

    def call(self, y_true, y_pred):

        return sparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred,
                                             class_weight=self.class_weight,
                                             gamma=self.gamma,
                                             from_logits=self.from_logits)

import tensorflow.keras.backend as K

class Seg_loss:
    def __init__(self, batch_size, use_aux=False, distribute_mode=True, use_focal=False):
        self.batch_size = batch_size
        self.alpha = 0.25
        self.gamma = 2.0
        self.num_classes = 19
        self.distribute_mode = distribute_mode
        self.use_focal = use_focal

        if use_aux:
            self.aux_factor = 0.4
        else:
            self.aux_factor = 1

        # self.class_weight = {0: 0.8373, 1: 0.918, 2: 0.866, 3: 1.0345,
        #                 4: 1.0166, 5: 0.9969, 6: 0.9754, 7: 1.0489,
        #                 8: 0.8786, 9: 1.0023, 10: 0.9539, 11: 0.9843,
        #                 12: 1.1116, 13: 0.9037, 14: 1.0865, 15: 1.0955,
        #                 16: 1.0865, 17: 1.1529, 18: 1.0507}

        self.class_weight = (0.8373, 0.918, 0.866, 1.0345,
                        1.0166, 0.9969, 0.9754, 1.0489,
                        0.8786, 1.0023, 0.9539, 0.9843,
                        1.1116, 0.9037, 1.0865, 1.0955,
                        1.0865, 1.1529, 1.0507)

        self.cls_weight = tf.constant(self.class_weight, tf.float32)



    def ce_loss(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=3)
        y_true = tf.reshape(y_true, [-1,])
        # todo
        raw_prediction = tf.reshape(y_pred, [-1, self.num_classes])
        indices = tf.squeeze(tf.where(tf.less_equal(y_true, self.num_classes-1)), 1)
        gt = tf.cast(tf.gather(y_true, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)


        if self.distribute_mode:
            ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)(y_true=gt,
                                                                                                          y_pred=prediction)


            # ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=prediction)

            weights = tf.gather(self.cls_weight, gt)
            ce_loss = (ce_loss * weights) * self.aux_factor
            # ce_loss *= self.cls_weight

        else:
            ce_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=prediction))

        # total_loss = ce_loss * self.aux_factor

        return ce_loss

    def focal_loss(self, y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=3)
        y_true = tf.reshape(y_true, [-1,])

        raw_prediction = tf.reshape(y_pred, [-1, self.num_classes])
        indices = tf.squeeze(tf.where(tf.less_equal(y_true, self.num_classes-1)), 1)
        gt = tf.cast(tf.gather(y_true, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)
        prob = tf.nn.softmax(prediction, axis=-1)

        if self.distribute_mode:
            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=prediction)

        else:
            ce_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=prediction)

        y_true_rank = gt.shape.rank
        probs = tf.gather(prob, gt, axis=-1, batch_dims=y_true_rank)

        focal_modulation = (1 - probs) ** self.gamma
        fl_loss = (focal_modulation * ce_loss) * self.aux_factor
        loss = tf.reduce_mean(fl_loss)

        # y_true = tf.squeeze(y_true, axis=-1)
        # probs = tf.nn.softmax(y_pred, axis=-1)
        # ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
        #                                                      reduction=tf.keras.losses.Reduction.NONE)(y_true=y_true, y_pred=y_pred)
        #
        # y_true_rank = y_true.shape.rank
        # probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)
        #
        # focal_modulation = (1 - probs) ** self.gamma
        # loss = focal_modulation * ce #* self.alpha
        # loss = K.mean(fl_loss)


        return loss

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha= 0.25
    y_true = tf.squeeze(y_true, -1)

    probs = tf.nn.softmax(y_pred, axis=-1)

    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=y_pred,
    )

    y_true_rank = y_true.shape.rank
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)

    focal_modulation = (1 - probs) ** gamma
    fl_loss = focal_modulation * xent_loss * alpha

    loss = tf.reduce_mean(fl_loss)


    return loss

def ce_loss(y_true, y_pred):

    y_true = tf.squeeze(y_true, -1)



    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y_true,
        logits=y_pred,
    )


    loss = tf.reduce_mean(xent_loss)

    return loss
