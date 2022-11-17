# -*- coding: utf-8 -*-
"""Transformers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-myYN08swe09digC2GtU3n-hp2V6OGN6
"""

!pip install transformers[sentencepiece]

!pip install datasets

from datasets import load_dataset
dataset = load_dataset('SergiiGurbych/sent_anal_ukr_binary')
dataset = dataset["train"]

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("youscan/ukr-roberta-base")

tokenized_data = tokenizer(dataset['text'], return_tensors="np", padding=True)
# it is of the type <class 'transformers.tokenization_utils_base.BatchEncoding'>
# Keras will not accept a BatchEncoding
#I need to convert it into a dictionary by adding a .data at the end

tokenized_data1=tokenized_data.data
# В получившемся словаре два ключа: dict_keys(['input_ids', 'attention_mask'])
tokenized_data2=tokenized_data['input_ids']

import numpy as np
labels = np.array(dataset["labels"])
labels=np.reshape(labels, (201,1))

from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
# Load and compile our model
model = TFAutoModelForSequenceClassification.from_pretrained("youscan/ukr-roberta-base", from_pt=True, num_labels=2)

model.compile(optimizer=Adam(learning_rate=3e-5), loss='binary_crossentropy')

model.fit(tokenized_data2, labels, epochs=10, verbose=True, validation_split=0.2)

# Тестируем на примерах
text = "Ця потвора вкрала в неї капелюха"
text1="Я тебе кохаю, бо ти найкрасивіша у світі, і я завжди буду любити тільки тебе"
encoding = tokenizer(text, return_tensors="np", padding=True)['input_ids']
encoding1 = tokenizer(text1, return_tensors="np", padding=True)['input_ids']
#print(encoding, encoding1)

outputs = model.predict(encoding1)
class_preds = np.argmax(outputs["logits"])
print(outputs)
print(class_preds)

'''
model.save('model_sent_anal_ukr1')

!zip -r model_sent_anal_ukr1.zip model_sent_anal_ukr1

from google.colab import files
files.download("model_sent_anal_ukr1.zip")
'''
