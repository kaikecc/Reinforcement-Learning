import tensorflow as tf
from tensorflow.keras.metrics import Metric

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        true_positives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_positives = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        false_negatives = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))

        self.true_positives.assign_add(tf.reduce_sum(tf.cast(true_positives, self.dtype)))
        self.false_positives.assign_add(tf.reduce_sum(tf.cast(false_positives, self.dtype)))
        self.false_negatives.assign_add(tf.reduce_sum(tf.cast(false_negatives, self.dtype)))

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_states(self):
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))
