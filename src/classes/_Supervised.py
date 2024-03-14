from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
import logging

class Supervised:
    def __init__(self, dataset_train_scaled, dataset_test_scaled):
        
        # Ajuste na definição de num_classes e input_shape
        self.input_shape = dataset_train_scaled.shape[1] - 1  # Correto como uma dimensão única
        self.x_train, self.y_train = dataset_train_scaled[:, :-1], dataset_train_scaled[:, -1]
        self.x_test, self.y_test = dataset_test_scaled[:, :-1], dataset_test_scaled[:, -1]
        self.x_train = self.x_train.astype('float32')
        self.y_train = self.y_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.y_test = self.y_test.astype('float32')

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
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        # Treina o modelo
        model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        return model

    def keras_evaluate(self, model):
        score = model.evaluate(self.x_test, self.y_test, verbose=0)

        # Correção na forma de usar o logging para registrar a perda do teste
        logging.info(f"Test loss: {score[0]}")
        print(f"Test accuracy: {score[1]}")
