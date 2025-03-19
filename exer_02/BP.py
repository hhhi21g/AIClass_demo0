import numpy as np
from sklearn.preprocessing import OneHotEncoder


def load_data(fileName):
    data = np.loadtxt(fileName)
    features = data[:, :-1]
    labels = data[:, -1].astype(int)
    return features, labels.reshape((len(labels), 1))


# 初始化权重及偏置
def init_param(num_inputs, num_hiddens, num_outputs):
    w1 = np.random.randn(num_inputs, num_hiddens) * 0.01
    b1 = np.zeros((1, num_hiddens))
    w2 = np.random.randn(num_hiddens, num_outputs) * 0.01
    b2 = np.zeros((1, num_outputs))
    return w1, b1, w2, b2


def encode_y(y):
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)
    return y_onehot


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止溢出
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# 前向传播
def forward(X, w1, b1, w2, b2):
    Z1 = X @ w1 + b1
    A1 = sigmoid(Z1)
    Z2 = A1 @ w2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# 反向传播
def backward(X, y, Z1, A1, A2, w2):
    m = X.shape[0]
    dZ2 = A2 - y
    dw2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m

    dA1 = dZ2 @ w2.T
    A1 = np.asarray(A1)
    dA1 = np.asarray(dA1)
    dZ1 = dA1 * (A1 * (1 - A1))
    dw1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m
    return dw1, db1, dw2, db2


def train(X, y, num_hiddens, lr=0.01, epochs=20000):
    num_inputs = X.shape[1]
    num_outputs = y.shape[1]

    w1, b1, w2, b2 = init_param(num_inputs, num_hiddens, num_outputs)

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward(X, w1, b1, w2, b2)
        dw1, db1, dw2, db2 = backward(X, y, Z1, A1, A2, w2)

        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

        if epoch % 1000 == 0:
            loss = -np.mean(y * np.log(A2 + 1e-8))
            print(f"Epoch {epoch}, Loss {loss:.4f}")

    return w1, b1, w2, b2


def predict(X, w1, b1, w2, b2):
    _, _, _, A2 = forward(X, w1, b1, w2, b2)
    return np.argmax(A2, axis=1)


def accuracy(preds, test_labels):
    test_labels = test_labels.flatten()
    preds = preds.flatten()
    return np.mean(test_labels == preds)


if __name__ == "__main__":
    train_features, train_labels = load_data("Iris-train.txt")
    test_features, test_labels = load_data("Iris-test.txt")

    # 返回每个样本属于每个种类的概率，这里使用独热编码处理
    train_labels_onehot = encode_y(train_labels)

    num_inputs, num_outputs, num_hiddens = train_features.shape[1], 3, 10

    w1, b1, w2, b2 = train(train_features, train_labels_onehot, num_hiddens)

    preds = predict(test_features, w1, b1, w2, b2)

    acc = accuracy(preds, test_labels)
    print(f"准确率: {acc * 100:.2f}%")
