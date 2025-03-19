import numpy as np
from sklearn.preprocessing import OneHotEncoder


def load_data(fileName):
    data = np.loadtxt(fileName)
    features = data[:, :-1]
    labels = data[:, -1]
    return features, labels.reshape((len(labels), 1))


# 初始化权重及偏置
def init_param(num_inputs, num_hiddens, num_outputs):
    w1 = np.random.randn(num_inputs, num_hiddens) * 0.01
    b1 = np.zeros((1, num_hiddens))
    w2 = np.random.randn(num_hiddens, num_outputs) * 0.01
    b2 = np.zeros((1, num_outputs))
    return w1, b1, w2, b2


def encode_y(y):
    encoder = OneHotEncoder()
    y_onehot = encoder.fit_transform(y)
    return y_onehot


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# 向前传播
def forward(X, w1, b1, w2, b2):
    Z1 = X @ w1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ w2 + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def backward(X,y,Z1,A1,A2,W2):
    


def train(X, y, num_hiddens, lr=0.1, epochs=10):
    num_inputs = X.shape[1]
    num_outputs = y.shape[1]
    w1, b1, w2, b2 = init_param()

    for epoch in range(epochs):
        Z1,A1,Z2,A2 = forward(X,w1,b1,w2,b2)


if __name__ == "__main__":
    train_features, train_labels = load_data("Iris-train.txt")
    test_features, test_labels = load_data("Iris-test.txt")

    # 返回每个样本属于每个种类的概率，这里使用独热编码处理
    train_labels_onehot = encode_y(train_labels)

    num_inputs, num_outputs, num_hiddens = train_features.shape[1], 3, 10
    w1, b1, w2, b2 = init_param(num_inputs, num_hiddens, num_outputs)
