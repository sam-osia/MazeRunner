import tensorflow as tf


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
    button_map = {'up': [1, 0, 0, 0], 'right': [0, 1, 0, 0], 'down': [0, 0, 1, 0], 'left': [0, 0, 0, 1]}
    return button_map[button]


X, Y = get_data()

print(X)
print(Y)
