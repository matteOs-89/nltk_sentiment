#!/usr/bin/env python
# coding: utf-8

# In[40]:


import nltk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


nltk.download("vader_lexicon")


# In[10]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


# For this task we will use NLTK framework to implyment Sentiment Analysis on Amazon review dataset. 
# 
# NLTK provides tools that enables us to carry out Sentiment Analysis in an efficient manner. Tools such as Vader's SentimentIntensityAnalyzer is available and we will explore it further in this project.

# In[4]:


sid = SentimentIntensityAnalyzer()


# In[12]:


df = pd.read_csv("/Users/eseosa/Desktop/NLP/UPDATED_NLP_COURSE/TextFiles/amazonreviews.tsv", sep="\t")


# In[14]:


print(df.head())


# In[17]:


df["label"].value_counts()

# neg    5097
# pos    4903


# In[20]:


df.dropna(inplace=True)


# In[25]:


space_index = []

for idx,label,review in df.itertuples():
    
    if type(review) == str:
        if review.isspace():
            space_index.append(review)
            
space_index


# In[27]:


print(df.iloc[0:6]["review"])


# In[31]:


for i in range(6):
    
    print(sid.polarity_scores(df.iloc[i]["review"]))


# In[34]:


df["scores"] = df["review"].apply(lambda review: sid.polarity_scores(review))
df["compound"] = df["scores"].apply(lambda idx: idx["compound"] )


# In[38]:


df["sentiment"] = df["compound"].apply(lambda x: "pos" if x >=0 else "neg")


# In[42]:


print(df)


# In[53]:


print(accuracy_score(df['label'], df["sentiment"]))
 
# 70.9% 

"""
Some elements that could effect out model perfomance in includes:

sarcastic reviews
spelling difficulties
language barriers

"""

print(classification_report(df['label'], df["sentiment"]))


# In[45]:


print(confusion_matrix(df['label'], df["sentiment"]))


# In[54]:


def sentiment_rating(comment):
    
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(comment)
    
    if scores["compound"]> 0:
        print("Positive Review")
        
    elif scores["compound"] == 0:
        print("Neutral Review")
        
    else:
        print("Review is Negative")


# In[59]:


sentiment_rating('poor service')


# In[56]:


sentiment_rating('great service')


# In[ ]:




