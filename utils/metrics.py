import tensorflow as tf

class MeanIOU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.squeeze(y_true, -1)
        y_pred = tf.argmax(y_pred, axis=-1)

        return super().update_state(y_true, y_pred, sample_weight)