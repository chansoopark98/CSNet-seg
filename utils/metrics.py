import tensorflow as tf

class MeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_pred = tf.argmax(y_pred, axis=-1)
        # y_true = tf.squeeze(y_true, -1)

        # indices = y_true > 0

        # y_true = tf.boolean_mask(y_true, indices)
        # y_pred = tf.boolean_mask(y_pred, indices)

        y_pred = tf.argmax(y_pred, axis=-1)
        # y_pred = tf.squeeze(y_pred, axis=-1)

        y_true = tf.squeeze(y_true, axis=-1)

        indices = tf.cast(tf.where(tf.equal(y_true, 0), 0, 1), tf.int64)

        y_true *= indices
        y_pred *= indices

        return super().update_state(y_true, y_pred, sample_weight)


class MIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.squeeze(y_true, -1)

        return (super().update_state(y_true, y_pred, sample_weight))+0.1