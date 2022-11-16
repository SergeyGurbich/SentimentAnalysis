# SentimentAnalysis
dataset_sent_anal_ukr_upd is a marked dataset for the Ukrainian language. 
It consists of sentences marked 0, 1 or 2 for negative, neutral or positive mode. 
The dataset is based on the classic text Shadows of Forgotten Ancestors written by Mykhailo Kotsiubynsky. 
The markup of the sentences was done automatically with the uploaded script Sent_Anal_Ukr_toDataset.py
based on the lists of positive and negative words from the Sentiment Lexicons for All Major Languages project (Chen & Skiena, ACL 2014). 
These lists were checked and edited manually by me to exclude ambiguous and mistakenly included words.

dataset_sent_anal_ukr_manual is a manually written and labeled dataset, it consists of 201 sentences marked 0 or 1 for negative or positive mode.

Sent_Anal_Ukr_train.py is a script for training a Ukrainian language model to make a binary sentence classification. 

