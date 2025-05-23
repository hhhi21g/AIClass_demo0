# ✅ 完整版（修正）：使用 roberta-large-mnli 并正确加载分类模型结构

import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AutoModelForSequenceClassification
from tqdm import tqdm
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型本地路径
model_name = 'FacebookAI/roberta-large-mnli'

# Dataset 定义
class MyDataset(Dataset):
    def __init__(self, data_list, test_mode=False):
        self.data = data_list
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        premise = self.data[idx][0]
        hypothesis = self.data[idx][1]
        encoding = self.tokenizer(premise, hypothesis, add_special_tokens=True, max_length=196,
                                  padding='max_length', truncation=True, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        if self.test_mode:
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            label = torch.tensor(self.data[idx][2], dtype=torch.long)
            return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}

# 数据加载器
class DataProcessor():
    def __init__(self, path):
        self.data_df = pd.read_csv(path)

    def get_dataloader(self, batch_size):
        train_df, val_df = train_test_split(self.data_df, test_size=0.2, random_state=42)
        train_data = train_df[['premise', 'hypothesis', 'label']].values.tolist()
        val_data = val_df[['premise', 'hypothesis', 'label']].values.tolist()
        return (
            DataLoader(MyDataset(train_data), batch_size=batch_size, shuffle=True),
            DataLoader(MyDataset(val_data), batch_size=batch_size // 2)
        )

# 提前停止
class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_accuracy = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_accuracy):
        if self.best_accuracy is None:
            self.best_accuracy = val_accuracy
        elif val_accuracy > self.best_accuracy + self.delta:
            self.best_accuracy = val_accuracy
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# 训练准备
train_path = "D:/AIClass_demo/AIClass_demo0/single_work/dataSet/train_augmented.csv"
processor = DataProcessor(train_path)
batch_size = 16
epochs = 10
train_dataloader, dev_dataloader = processor.get_dataloader(batch_size)

# 加载模型
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

# 解冻最后4层
for name, param in model.base_model.named_parameters():
    if any(f'layer.{i}' in name for i in [20, 21, 22, 23]):
        param.requires_grad = True
    else:
        param.requires_grad = False

# 优化器和调度器
optimizer = AdamW([
    {'params': [p for n, p in model.named_parameters() if 'classifier' in n], 'lr': 1e-4},
    {'params': [p for n, p in model.named_parameters() if any(f'layer.{i}' in n for i in [20,21,22,23])], 'lr': 1e-5}
])
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
early_stopping = EarlyStopping(patience=3)

# 训练
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}/{epochs}'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}")

    model.eval()
    val_loss = 0
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc='Validating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            preds += torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            true_labels += labels.cpu().tolist()
    accuracy = accuracy_score(true_labels, preds)
    print(f'Validation Loss: {val_loss / len(dev_dataloader):.4f}, Accuracy: {accuracy:.4f}')

    early_stopping(accuracy)
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# 推理
model.eval()
test_data = pd.read_csv('D:/AIClass_demo/AIClass_demo0/single_work/dataSet/test.csv')
test_list = test_data[['premise', 'hypothesis']].values.tolist()
test_dataset = MyDataset(test_list, test_mode=True)
test_loader = DataLoader(test_dataset, batch_size=8)
test_probs = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        test_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

avg_probs = np.concatenate(test_probs, axis=0)
np.save("D:/AIClass_demo/AIClass_demo0/single_work/dataSet/roberta_large_mnli_softmax.npy", avg_probs)
final_preds = np.argmax(avg_probs, axis=1)

# 保存结果
submission_df = pd.DataFrame({
    'id': test_data['id'],
    'prediction': final_preds
})
submission_df.to_csv('D:/AIClass_demo/AIClass_demo0/single_work/dataSet/submission.csv', index=False)
print("✅ 单模型训练+推理完成，结果已保存")
