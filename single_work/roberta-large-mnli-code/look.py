import numpy as np

# 加载 .npy 文件
probs = np.load('D:\\AIClass_demo\\AIClass_demo0\\single_work\\dataSet\\roberta_large_mnli_softmax.npy')

# 查看数组维度（行数 = 样本数，列数 = 类别数）
print("📐 维度信息：", probs.shape)

# 查看前5个样本的 softmax 概率
print("🔍 前5个样本的概率分布：")
print(probs[:5])

# 如果需要查看某个样本的预测类别：
preds = np.argmax(probs, axis=1)
print("✅ 对应预测标签：", preds[:5])
