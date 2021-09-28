import tensorflow as tf

class MeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)

        y_true = tf.squeeze(y_true, axis=3)
        y_true = tf.reshape(y_true, [-1,])

        raw_prediction = tf.reshape(y_pred, [-1,])
        indices = tf.squeeze(tf.where(tf.less_equal(y_true, self.num_classes-1)), 1)
        y_true = tf.cast(tf.gather(y_true, indices), tf.int32)
        y_pred = tf.gather(raw_prediction, indices)


        # y_pred = tf.argmax(y_pred, axis=-1)
        # y_true = tf.squeeze(y_true, axis=-1)
        #
        # indices = tf.cast(tf.where(tf.equal(y_true, 0), 0, 1), tf.int32)
        #
        # y_true *= indices
        # y_pred *= indices

        return super().update_state(y_true, y_pred, sample_weight)


class MIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.squeeze(y_true, -1)

        return (super().update_state(y_true, y_pred, sample_weight))+0.1