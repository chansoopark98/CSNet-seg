import tensorflow as tf

class Loss:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def sparse_categorical_loss(self, y_true, y_pred):
        ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                           reduction=tf.keras.losses.Reduction.NONE)(y_true=y_true,
                                                                                                     y_pred=y_pred)
        return ce

    def aux_loss(self, y_true, y_pred):
        aux_logits = y_pred[:, :, :0]
        logits = y_pred[:, :, 0:]

        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)(y_true=y_true,
                                                                                                          y_pred=logits)

        aux_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction=tf.keras.losses.Reduction.NONE)(y_true=y_true,
                                                                                                          y_pred=aux_logits)

        total_loss = ce_loss + (aux_loss * 0.4)

        return total_loss
