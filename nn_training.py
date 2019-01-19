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
    return X, Y


X, Y = get_data()

print(X)
print(Y)
