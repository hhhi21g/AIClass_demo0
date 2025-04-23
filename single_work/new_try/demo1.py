import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from transformers import AutoConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
model_name = 'D:\\AIClass_demo\\AIClass_demo0\\single_work\\bert-base-multilingual-uncased'
# model_name = 'D:\\AIClass_demo\\AIClass_demo0\\single_work\\'

# model_name = 'bert-base-uncased'


# cache_dir = r'C:\Users\a1824\.cache\huggingface\hub\models--bert-base-uncased'

# 覆写torch.utils.data下的Dataset类，必不可少
class MyDataset(Dataset):
    # 数据封装，出去的满足神经网络训练的张量形式
    def __init__(self, data_list, test_mode=False):  # data_list = [[premise, hypothesis, label],[]...]
        self.data = data_list
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data[idx][0]
        hypothesis = self.data[idx][1]

        encoding = self.tokenizer(premise, hypothesis,
                                  add_special_tokens=True,  # 会在两个句子之间加一个特殊token
                                  max_length=128,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        if self.test_mode:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        else:
            label = self.data[idx][2]
            label = torch.tensor(label, dtype=torch.long)
            return {'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'label': label}


class DataProcessor():
    def __init__(self, path):
        self.data_df = pd.read_csv(path)  # 加载数据

    # EDA
    def eda(self, opt):
        assert opt in ['view_data', 'count_class_distri', 'text_length_distri'], 'opt is wrong'
        if opt == 'view_data':  # 查看数据前几行
            self._view_data()
        elif opt == 'count_class_distri':  # 查看类别标签的分布情况:0蕴含，1不相干，2矛盾
            self._count_class_distri()
        elif opt == 'text_length_distri':  # 查看文本长度的分布情况
            self._text_length_distri()

    # 获取数据封装(训练和开发认证)
    def get_dataloader(self):
        data_list = self.data_df[['premise', 'hypothesis', 'label']].values.tolist()
        random.shuffle(data_list)
        k = len(data_list) // 5

        train_dataset = MyDataset(data_list[k:])
        train_dataloader = DataLoader(train_dataset, batch_size=8)
        dev_dataset = MyDataset(data_list[:k])
        dev_dataloader = DataLoader(dev_dataset, batch_size=16)
        return train_dataloader, dev_dataloader

    def _view_data(self):  # "_"表示供EDA调用的辅助方法
        print(self.data_df.head())

    # 统计不同label有多少个，绘图表示
    def _count_class_distri(self):
        plt.figure(figsize=(6, 4))
        sns.countplot(data=self.data_df, x='label')
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.show()

    def _text_length_distri(self):
        self.data_df['premise_length'] = self.data_df['premise'].apply(lambda x: len(x.split()))
        self.data_df['hypothesis_length'] = self.data_df['hypothesis'].apply(lambda x: len(x.split()))

        # 创建一个带有子图的画布
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

        # 在第一个子图中绘制premise的箱型图
        sns.boxplot(data=self.data_df, y='premise_length', ax=axes[0])
        axes[0].set_title('Premise Length Distribution')
        axes[0].set_ylabel('Length')

        # 在第二个子图中绘制hypothesis的箱型图
        sns.boxplot(data=self.data_df, y='hypothesis_length', ax=axes[1])
        axes[1].set_title('Hypothesis Length Distribution')
        axes[1].set_ylabel('Length')

        # 调整子图间距
        plt.tight_layout()
        plt.show()


processor = DataProcessor('..\\data\\augmented_train.csv')
processor.eda('view_data')
processor.eda('count_class_distri')
processor.eda('text_length_distri')
train_dataloader, dev_dataloader = processor.get_dataloader()


# for batch in train_dataloader:
#     print(batch['input_ids'].shape)  # [8,128],[batch_size,max_length]
#     print(batch['attention_mask'].shape)  # [8,128]
#     print(batch['label'].shape)  # [8]
#     break
class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW

config = AutoConfig.from_pretrained(model_name, num_labels=3)
# config.hidden_dropout_prob = 0.4

model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5,weight_decay=0.01)

import torch

# 设置训练轮次
epochs = 10
early_stopping = EarlyStopping(patience=8, delta=0.001)

# 创建学习率调度器
# scheduler = get_linear_schedule_with_warmup(optimizer,
#                                             num_warmup_steps=0,
#                                             num_training_steps=len(train_dataloader)*epochs)

for epoch in range(epochs):
    model.train()  # 设置模型为训练模式
    total_loss = 0
    loop = tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}/{epoch}')
    for batch in loop:
        optimizer.zero_grad()  # 清除上一轮的梯度

        # 获取批次数据
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # 更新模型参数
        # scheduler.step()
        loop.set_postfix(loss=loss.item())
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_dataloader)}')

    # 假设你有一个验证集的 DataLoader（val_dataloader）
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        loop = tqdm(dev_dataloader, desc='Validating')
        for batch in loop:  # 验证集的数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    avg_val_loss = val_loss / len(dev_dataloader)
    print(f'Validation Loss: {avg_val_loss}, Accuracy: {accuracy}')

    early_stopping(avg_val_loss)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

test_data = pd.read_csv('..\\data\\test.csv')

test_list = test_data[['premise', 'hypothesis']].values.tolist()

test_dataset = MyDataset(test_list, test_mode=True)
test_dataloader = DataLoader(test_dataset, batch_size=16)

model.eval()
predictions = []

with torch.no_grad():  # 禁用梯度计算
    loop = tqdm(test_dataloader, desc='Predicting')
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        # 预测
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

        predictions.extend(preds)

# 创建与上传的格式一致的结果 DataFrame
submission_df = pd.DataFrame({
    'id': test_data['id'],  # 使用测试数据的 id 列
    'prediction': predictions
})

# 保存结果为新的 CSV 文件
submission_df.to_csv('submission.csv', index=False)
print("预测完成")
