import pandas as pd
from collections import Counter

# 加载三个预测结果
df1 = pd.read_csv("submission (3).csv")
df2 = pd.read_csv("submission (4).csv")
df3 = pd.read_csv("submission (5).csv")

# 确保按 id 对齐
# df1 = df1.sort_values('id').reset_index(drop=True)
# df2 = df2.sort_values('id').reset_index(drop=True)
# df3 = df3.sort_values('id').reset_index(drop=True)

# 比较差异
differences = (df1['prediction'] != df2['prediction']) | \
              (df1['prediction'] != df3['prediction']) | \
              (df2['prediction'] != df3['prediction'])

print(f"⚖️ 三个文件中 prediction 完全一致的行数：{(~differences).sum()}")
print(f"🔀 prediction 有至少一个不一致的行数：{differences.sum()}")

# 投票融合
final_preds = []
for p1, p2, p3 in zip(df1['prediction'], df2['prediction'], df3['prediction']):
    vote = Counter([p1, p2, p3]).most_common(1)[0][0]
    final_preds.append(vote)

# 生成最终融合结果
submission = pd.DataFrame({
    'id': df1['id'],
    'prediction': final_preds
})
submission.to_csv("submission_vote.csv", index=False)
print("✅ 投票融合后的 submission_vote.csv 已保存")
