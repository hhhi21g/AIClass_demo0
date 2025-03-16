import numpy as np
import pandas as pd
from math import log2
from graphviz import Digraph

# 读取数据
def load_data(fileName):
    with open(fileName, "r") as file:
        lines = file.readlines()[1:-1]
    data = np.loadtxt(lines, dtype=float)  # 以制表符分隔
    features = data[:, :-1]  # 取所有行的前 N-1 列作为特征
    labels = data[:, -1].astype(int)  # 取最后一列作为类别，并转换为整数
    return features, labels

# 计算熵
def entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    prob = counts / len(labels)
    return -sum(p * log2(p) for p in prob if p > 0)  # 避免 log(0)

# 计算最佳划分点（适用于连续变量）
def best_split(feature, labels):
    sorted_indices = np.argsort(feature)  # 按特征值排序
    sorted_feature = feature[sorted_indices]
    sorted_labels = labels[sorted_indices]

    best_gain_ratio = -1
    best_threshold = None

    for i in range(1, len(sorted_feature)):  # 计算所有可能的划分点
        threshold = (sorted_feature[i] + sorted_feature[i - 1]) / 2
        left_labels = sorted_labels[:i]
        right_labels = sorted_labels[i:]

        # 计算信息增益
        H = entropy(labels)
        H_left = entropy(left_labels)
        H_right = entropy(right_labels)

        weight_left = len(left_labels) / len(labels)
        weight_right = len(right_labels) / len(labels)
        gain = H - (weight_left * H_left + weight_right * H_right)

        # 计算增益率（避免除零错误）
        split = - (weight_left * log2(weight_left) + weight_right * log2(weight_right)) if weight_left > 0 and weight_right > 0 else 0
        gain_ratio = gain / split if split != 0 else 0

        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_threshold = threshold

    return best_threshold, best_gain_ratio

# 选择最佳特征
def choose_best_feature(features, labels):
    n_features = features.shape[1]
    best_feature_idx = -1
    best_threshold = None
    best_gain_ratio = -1

    for i in range(n_features):
        threshold, gain_ratio = best_split(features[:, i], labels)
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature_idx = i
            best_threshold = threshold

    return best_feature_idx, best_threshold

# 递归构建决策树
def build_tree(features, labels, feature_names, depth=0, max_depth=10):
    # 终止条件 1：只有一个类别，直接返回该类别
    if len(np.unique(labels)) == 1:
        return {'label': str(labels[0])}

    # 终止条件 2：达到最大深度
    if depth >= max_depth:
        majority_label = str(np.argmax(np.bincount(labels)))
        return {'label': majority_label}

    # 选择最佳划分特征
    best_feature_idx, best_threshold = choose_best_feature(features, labels)
    if best_feature_idx == -1:  # 处理无法划分的情况
        majority_label = str(np.argmax(np.bincount(labels)))
        return {'label': majority_label}

    best_feature_name = feature_names[best_feature_idx]
    tree = {f"{best_feature_name} ≤ {best_threshold:.2f}": {}}

    left_indices = features[:, best_feature_idx] <= best_threshold
    right_indices = features[:, best_feature_idx] > best_threshold

    # 终止条件 3：避免子集为空
    if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
        majority_label = str(np.argmax(np.bincount(labels)))
        return {'label': majority_label}

    # 递归构建左右子树
    tree[f"{best_feature_name} ≤ {best_threshold:.2f}"]['<='] = build_tree(features[left_indices], labels[left_indices], feature_names, depth+1, max_depth)
    tree[f"{best_feature_name} ≤ {best_threshold:.2f}"]['>'] = build_tree(features[right_indices], labels[right_indices], feature_names, depth+1, max_depth)

    return tree

# 分类预测
def classify(tree, sample, feature_names):
    if 'label' in tree:
        return tree['label']

    feature = list(tree.keys())[0]
    feature_name, threshold = feature.split(" ≤ ")
    feature_idx = np.where(feature_names == feature_name)[0][0]
    threshold = float(threshold)

    if sample[feature_idx] <= threshold:
        subtree = tree[feature]['<=']
    else:
        subtree = tree[feature]['>']

    return classify(subtree, sample, feature_names)

# 计算准确率
def accuracy(tree, test_features, test_labels, feature_names):
    correct = sum(
        classify(tree, test_features[i], feature_names) == str(test_labels[i])
        for i in range(len(test_labels))
    )
    return correct / len(test_labels)

# 决策树可视化
def visualize_tree(tree, dot=None):
    if dot is None:
        dot = Digraph()

    root = list(tree.keys())[0]
    dot.node(root)

    for relation, subtree in tree[root].items():
        if 'label' in subtree:
            child = f"{root}_{relation}_{subtree['label']}"
            dot.node(child, subtree['label'])
            dot.edge(root, child, label=relation)
        else:
            child = list(subtree.keys())[0]
            dot.node(child)
            dot.edge(root, child, label=relation)
            visualize_tree(subtree, dot)

    return dot

# 递归将决策树保存为文本格式
def save_tree_to_txt(tree, file, indent=""):
    if "label" in tree:
        file.write(f"{indent}Label: {tree['label']}\n")
        return
    feature = list(tree.keys())[0]
    file.write(f"{indent}{feature}:\n")
    save_tree_to_txt(tree[feature]["<="], file, indent + "  ")
    save_tree_to_txt(tree[feature][">"], file, indent + "  ")

# 预测测试集并保存结果
def save_predictions_to_txt(tree, test_features, test_labels, feature_names, file_name="predictions.txt"):
    with open(file_name, "w") as f:
        for i in range(len(test_features)):
            prediction = classify(tree, test_features[i], feature_names)  # 预测结果
            true_label = test_labels[i]  # 真实标签
            f.write(f"Sample {i+1}: True Label = {true_label}, Predicted = {prediction}\n")

if __name__ == "__main__":
    feature_names = np.array(["F1", "F2", "F3", "F4"])  # 假设有 4 个特征

    train_features, train_labels = load_data("traindata.txt")
    test_features, test_labels = load_data("testdata.txt")

    tree = build_tree(train_features, train_labels, feature_names, max_depth=30)

    acc = accuracy(tree, test_features, test_labels, feature_names)
    print(f"分类准确率: {acc * 100:.2f}%")

    # 将决策树保存为 txt
    with open("decision_tree.txt", "w") as f:
        save_tree_to_txt(tree, f)

    print("决策树已保存至 decision_tree.txt")
    # 将测试结果保存到文件
    save_predictions_to_txt(tree, test_features, test_labels, feature_names)
    print("预测结果已保存至 predictions.txt")