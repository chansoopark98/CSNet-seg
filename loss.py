import tensorflow as tf
from tensorflow.keras.layers import Flatten

def ce_loss(y_true, y_pred):
    num_classes = 20
    y_pred = tf.argmax(y_pred, axis=3, output_type=tf.int32)
    y_pred = tf.cast(tf.expand_dims(y_pred, axis=3), dtype=tf.float32)
    print(y_pred)

    y_true = tf.one_hot(y_true, depth=3)



    # y_pred = tf.nn.softmax(y_pred)
    print("11")
    # tf.nn.log_softmax(y_pred)


    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    # loss = tf.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
    # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_true, logits=y_pred, pos_weight=2)

    return loss


def convert_to_logits(y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(),
                              1 - tf.keras.backend.epsilon())
    return tf.math.log(y_pred / (1 - y_pred))
beta = 0.25

def weighted_cross_entropyloss(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=3, output_type=tf.int32)
    y_pred = tf.cast(tf.expand_dims(y_pred, axis=3), dtype=tf.float32)
    y_pred = convert_to_logits(y_pred)
    # y_true = convert_to_logits(y_true)
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred,
                                                    labels=y_true,
                                                    pos_weight=pos_weight)

    return tf.reduce_mean(loss)