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

        y_true = tf.squeeze(y_true, axis=-1)
        y_true += 1
        y_true_mask = tf.where(tf.greater(y_true, 21), 0, 1)
        y_true_mask = tf.cast(y_true_mask, tf.int64)
        y_true = y_true * y_true_mask
        y_pred = tf.math.argmax(y_pred, axis=-1)
        y_pred += 1

        zeros_y_pred = tf.zeros(tf.shape(y_pred), tf.int64)
        zeros_y_pred += y_pred
        indices = tf.cast(tf.where(tf.equal(y_true, 0), 0, 1), tf.int64)

        y_true *= indices
        zeros_y_pred *= indices


        return super().update_state(y_true, y_pred, sample_weight)


class EdgeAccuracy(tf.keras.metrics.BinaryAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        edge_y_true = y_true
        edge_y_true = tf.cast(edge_y_true, tf.float32)

        grad_components = tf.image.sobel_edges(edge_y_true)
        grad_mag_components = grad_components ** 2
        grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)

        edge_y_true = tf.sqrt(grad_mag_square)

        edge_y_true = tf.clip_by_value(edge_y_true, 0, 1)

        return super().update_state(edge_y_true, y_pred, sample_weight)
