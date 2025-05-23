import pandas as pd

# 读取原始 CSV 文件
df = pd.read_csv('/kaggle/input/submit/submission (2).csv')

# 将数据保存为另一个 CSV 文件
df.to_csv('/kaggle/working/submission.csv', index=False)

print("finish")