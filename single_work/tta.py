import pandas as pd
import random
from nltk.corpus import wordnet

# 参数
tta_num = 5  # TTA组数（包括原文）

def synonym_replace(sentence, ratio=0.18):
    words = sentence.split()
    new_words = []
    for w in words:
        if random.random() < ratio:
            syns = wordnet.synsets(w)
            lemmas = [l.name().replace('_', ' ') for s in syns for l in s.lemmas() if l.name().lower() != w.lower()]
            if lemmas:
                new_word = random.choice(lemmas)
                new_words.append(new_word)
                continue
        new_words.append(w)
    return ' '.join(new_words)

# 读取测试集
test_df = pd.read_csv('dataSet\\test.csv')
test_premises = test_df['premise'].tolist()
test_hypos = test_df['hypothesis'].tolist()

tta_premises = []
tta_hypos = []

for p, h in zip(test_premises, test_hypos):
    premise_versions = [p]
    hypo_versions = [h]
    for i in range(tta_num - 1):
        premise_versions.append(synonym_replace(p))
        hypo_versions.append(synonym_replace(h))
    tta_premises.append(premise_versions)
    tta_hypos.append(hypo_versions)

# 保存为csv
tta_df = pd.DataFrame()
for t in range(tta_num):
    tta_df[f'premise_tta{t}'] = [ps[t] for ps in tta_premises]
    tta_df[f'hypo_tta{t}'] = [hs[t] for hs in tta_hypos]
tta_df['id'] = test_df['id']
tta_df.to_csv('test_tta_synonym.csv', index=False)
print("✅ TTA同义词增强文本已保存为 test_tta_synonym.csv")
