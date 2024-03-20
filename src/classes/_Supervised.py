import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import TensorBoard
import logging
import os
from classes._F1Score import F1Score

class Supervised:
    def __init__(self, path_save, dataset_train_scaled, dataset_test_scaled):
        
        # Ajuste na definição de num_classes e input_shape
        self.input_shape = dataset_train_scaled.shape[1] - 1  # Correto como uma dimensão única
        self.x_train, self.y_train = dataset_train_scaled[:, :-1], dataset_train_scaled[:, -1]
        self.x_test, self.y_test = dataset_test_scaled[:, :-1], dataset_test_scaled[:, -1]
        self.x_train = self.x_train.astype('float32')
        self.y_train = self.y_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.y_test = self.y_test.astype('float32')

        self.path_save = path_save

        # Aplicar one-hot encoding nos rótulos
        encoder = OneHotEncoder(sparse=False)
        self.y_train = encoder.fit_transform(self.y_train.reshape(-1, 1))
        self.y_test = encoder.transform(self.y_test.reshape(-1, 1))
        self.num_classes = self.y_train.shape[1]  # Número de classes baseado na codificação one-hot

    def keras_train(self, batch_size=32, epochs=2):
        # Define a arquitetura do modelo
        model = keras.Sequential([
            keras.Input(shape=self.input_shape),  # Correção aplicada aqui
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation="softmax")
        ])

        model.summary()

        # Compila o modelo
        #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()])


        # Prepara o callback do TensorBoard
        logdir_base = os.path.dirname(self.path_save)  # Sobe um nível (para '..\\models\\Abrupt Increase of BSW')
        logdir = os.path.join(logdir_base, 'tensorboard_logs')
        tensorboard_callback = TensorBoard(log_dir=logdir, histogram_freq=1)

        # Treina o modelo
        model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[tensorboard_callback])

        model.save(f'{self.path_save}_RNA')
        return model

    def keras_evaluate(self):
        # Carrega o modelo
        self.model = keras.models.load_model(f'{self.path_save}_RNA')
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        logging.info(f"Test loss: {score[0]}")
        logging.info(f"Test accuracy: {score[1]}")
        logging.info(f"Test precision: {score[2]}")
        logging.info(f"Test recall: {score[3]}")
        logging.info(f"Test F1 Score: {score[4]}")
        return score[1]
