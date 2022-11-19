# -*- coding: utf-8 -*-
"""
Script for fine-tuning the Transformer model "youscan/ukr-roberta-base"
on the dataset for sequence classification (polarity only)
for Ukrainian language 
"""

!pip install transformers[sentencepiece]
!pip install datasets

# load dataset and split it to train and validation (!) parts
from datasets import load_dataset
dataset = load_dataset('SergiiGurbych/sent_anal_ukr_binary')
dataset1 = dataset["train"]
dataset_sp=dataset1.train_test_split(test_size=0.2)

# Tokenize the splitted dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("youscan/ukr-roberta-base")

def tokenization(example):
    return tokenizer(example["text"])

dataset_tokenized = dataset_sp.map(tokenization, batched=True)

#Create two subsets from the tokenized dataset
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = dataset_tokenized["train"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    shuffle=True,
    batch_size=2,
    collate_fn=data_collator,
)

tf_validation_dataset = dataset_tokenized["test"].to_tf_dataset(
    columns=["input_ids", "attention_mask"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=2,
)

# Load and compile our model for Sequence Classification
from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

model = TFAutoModelForSequenceClassification.from_pretrained("youscan/ukr-roberta-base",
                                                              from_pt=True,
                                                              num_labels=2)
model.compile(optimizer=Adam(learning_rate=3e-5))#, loss=tf.keras.losses.BinaryCrossentropy)

# fit the model
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=3,
                                            restore_best_weights=True)
log=model.fit(tf_train_dataset,
    validation_data=tf_validation_dataset,
    epochs=8, 
    verbose=True,
    callbacks=[callback]
)

model.save_pretrained('model_sent_anal_ukr_binary_1')

model.summary()

# Checking the predicted output manually if needed
import numpy as np

text='''Сіре кошеня забило лапку і плаче.
Приємно вийти на прогулянку у безвітряний день.
Я запізнилась на автобус і мусила йти пішки, до того ж і прийшла на зустріч пізніше, ніж треба.
Якщо ти будеш увічливим з людьми, які можуть тобі допомогти, ти можеш бути набагато успішнішим / досягти набагато більше.
Мій колега потрапив в аварію: він ішов по тротуару і його збив велосипед.
Велосипедист зміг зіскочити з велосипеду, тому він не був поранений.
На щастя, мій колега не у важкому стані, але у нього болить рука.
Це адреса гарного місця, яке можна відвідати на вихідних, якщо тільки у друзів не зміняться плани.
Потім посеред вулиці зламалась вантажівка, тож рух повністю зупинився.
Сьогодні чудова погода і можна піти гуляти в парк, але ми як на зло маємо вчитися.'''
txt=text.split('\n')

for ele in txt:
  encoding = tokenizer(ele, return_tensors="np", padding=True)['input_ids']
  outputs = model4.predict(encoding)
  class_preds = np.argmax(outputs["logits"])
  if class_preds ==1:
    print(ele, '- позитивне! :)')
  elif class_preds ==0:
    print(ele, '- негативне! :(')
