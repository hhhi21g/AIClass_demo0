import pandas as pd

# 加载文件（改为你本地路径）
original_df = pd.read_csv("D:\\AIClass_demo\\AIClass_demo0\\single_work\\data\\train.csv")
augmented_df = pd.read_csv("D:\\AIClass_demo\\AIClass_demo0\\single_work\\data\\train_aug.csv")

# 基本信息
print(f"原始样本数: {len(original_df)}")
print(f"增强后样本数: {len(augmented_df)}")

# 检查增强后是否有不同于原始的 premise 文本
original_premises = set(original_df["premise"])
diff_df = augmented_df[~augmented_df["premise"].isin(original_premises)]

print(f"增强后不同于原始的样本数: {len(diff_df)}")
print("示例差异行：")
print(diff_df[["premise", "hypothesis", "label"]].head(5))
