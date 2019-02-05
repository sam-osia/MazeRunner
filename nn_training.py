import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle


def get_data():
    X = []
    Y = []
    with open('data.txt', 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    X = [num.split(',') for num in (line.split('\t')[0] for line in lines)]
    X = [[int(float(j)) for j in i] for i in X]
    Y = [line.split('\t')[1] for line in lines]
    Y = list(map(convert_to_number, Y))
    return X, Y


def convert_to_number(button):
    button_map = {'up': np.array([1, 0, 0, 0]), 'right': np.array([0, 1, 0, 0]), 'down': np.array([0, 0, 1, 0]), 'left': np.array([0, 0, 0, 1])}
    button_map = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
    return button_map[button]


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation=tf.nn.relu, input_shape=(37,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    X, Y = get_data()

    X = np.array(X)
    Y = np.array(Y)

    X, Y = shuffle(X, Y)

    X = tf.keras.utils.normalize(X, axis=1)

    len_of_data = int(len(X) * 0.8)
    print(len_of_data)
    X_train = X[:len_of_data]
    Y_train = Y[:len_of_data]
    X_test = X[len_of_data:]
    Y_test = Y[len_of_data:]

    model = create_model()
    model.fit(X_train, Y_train, epochs=200)

    model.save_weights('./model_weights')
    score, acc = (model.evaluate(X_test, Y_test))

    print('Test score:', score)
    print('Test accuracy', acc)
