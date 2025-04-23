import random
import pandas as pd
import nltk
from langdetect import detect, LangDetectException
from nltk.corpus import wordnet
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm
import concurrent.futures
import threading

# 确保已下载NLTK数据
# nltk.download('wordnet')

# 初始化线程锁
model_lock = threading.Lock()

# 加载翻译模型
model_name_trans = 'D:\\AIClass_demo\\AIClass_demo0\\single_work\\opus-mt-en-de'  # 英文->德文
model_trans = MarianMTModel.from_pretrained(model_name_trans).to('cuda')
tokenizer_trans = MarianTokenizer.from_pretrained(model_name_trans)

back_model_name = 'D:\\AIClass_demo\\AIClass_demo0\\single_work\\opus-mt-de-en'  # 德文->英文
back_model = MarianMTModel.from_pretrained(back_model_name).to('cuda')
back_tokenizer = MarianTokenizer.from_pretrained(back_model_name)

back_translation_log = []

def is_english(text):
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return False

def synonym_replacement(sentence):
    """替换随机单词为同义词"""
    words = sentence.split()
    if not words:
        return sentence
    word = random.choice(words)
    synonyms = wordnet.synsets(word)
    if synonyms:
        synonym = random.choice(synonyms).lemmas()[0].name()
        return sentence.replace(word, synonym, 1)
    return sentence


def back_translation_batch(sentences):
    """批量回译"""
    with model_lock:  # 使用线程锁保证线程安全
        # 英->德
        inputs = tokenizer_trans(sentences, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            'cuda')
        translated = model_trans.generate(**inputs)
        de_texts = tokenizer_trans.batch_decode(translated, skip_special_tokens=True)

        # 德->英
        back_inputs = back_tokenizer(de_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
            'cuda')
        back_translated = back_model.generate(**back_inputs)
        en_texts = back_tokenizer.batch_decode(back_translated, skip_special_tokens=True)

        for orig,bt in zip(sentences,en_texts):
            back_translation_log.append(f"{orig.strip()}|||{bt.strip()}")

    return en_texts


def random_deletion(sentence, p=0.1):
    """随机删除单词"""
    words = sentence.split()
    if len(words) < 5:
        return sentence
    return ' '.join([w for w in words if random.random() > p])


def augment_data(premise, hypothesis, label):
    """数据增强处理"""
    try:
        # 回译增强（50%概率执行）
        if random.random() < 0.7 and is_english(premise) and is_english(hypothesis):
            bt_results = back_translation_batch([premise, hypothesis])
            premise, hypothesis = bt_results[0], bt_results[1]

        # 同义词替换（40%概率执行）
        if random.random() < 0.5:
            premise = synonym_replacement(premise)
            hypothesis = synonym_replacement(hypothesis)

        # 随机删除（20%概率执行）
        if random.random() < 0.2:
            premise = random_deletion(premise)
            hypothesis = random_deletion(hypothesis)

    except Exception as e:
        print(f"Augmentation error: {e}")

    return premise, hypothesis, label


def save_augmented_data(input_csv, output_csv):
    """保存增强后的数据"""
    df = pd.read_csv(input_csv)
    augmented_data = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交所有任务
        futures = [executor.submit(augment_data, row['premise'], row['hypothesis'], row['label'])
                   for _, row in df.iterrows()]

        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Augmenting"):
            try:
                res = future.result()
                augmented_data.append(res)
            except Exception as e:
                print(f"Error processing: {e}")

    # 创建DataFrame并保存
    result_df = pd.DataFrame(augmented_data, columns=['premise', 'hypothesis', 'label'])
    result_df.to_csv(output_csv, index=False)
    print(f"Augmented data saved to {output_csv}")


# 使用示例
save_augmented_data('data/train.csv', 'data/augmented_train.csv')
