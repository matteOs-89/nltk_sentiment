
import nltk
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download("vader_lexicon")

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

"""
For this task we will use NLTK framework to implyment Sentiment Analysis on Amazon review dataset. 
NLTK provides tools that enables us to carry out Sentiment Analysis in an efficient manner. 
Tools such as Vader's SentimentIntensityAnalyzer is available and we will explore it further in this project.
"""

sid = SentimentIntensityAnalyzer()

df = pd.read_csv("/Users/eseosa/Desktop/NLP/UPDATED_NLP_COURSE/TextFiles/amazonreviews.tsv", sep="\t")

print(df.head())


df["label"].value_counts()

# neg    5097
# pos    4903


df.dropna(inplace=True)


""" Check and return index of space columns on our dataset"""

space_index = []

for idx,label,review in df.itertuples():
    
    if type(review) == str:
        if review.isspace():
            space_index.append(review)
            
space_index

print(df.iloc[0:6]["review"])


for i in range(6):
    
    print(sid.polarity_scores(df.iloc[i]["review"]))


df["scores"] = df["review"].apply(lambda review: sid.polarity_scores(review))

df["compound"] = df["scores"].apply(lambda idx: idx["compound"] )

df["sentiment"] = df["compound"].apply(lambda x: "pos" if x >=0 else "neg")


print(df)


print(accuracy_score(df['label'], df["sentiment"]))
 
# 70.9% 

"""
Some elements that could effect out model perfomance in includes:

sarcastic reviews
spelling difficulties
language barriers

"""

print(classification_report(df['label'], df["sentiment"]))

    """   
    precision    recall  f1-score   support

         neg       0.86      0.51      0.64      5097
         pos       0.64      0.91      0.75      4903

   micro avg       0.71      0.71      0.71     10000
   macro avg       0.75      0.71      0.70     10000
weighted avg       0.75      0.71      0.70     10000

"""

print(confusion_matrix(df['label'], df["sentiment"]))

"""
[[2623   2474]
 
 [ 435   4468]]
 
 """


def sentiment_rating(comment):
    
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(comment)
    
    if scores["compound"]> 0:
        print("Positive Review")
        
    elif scores["compound"] == 0:
        print("Neutral Review")
        
    else:
        print("Review is Negative")


sentiment_rating('poor service')

"""Neutral Review"""

sentiment_rating('great service')

"""Positive Review"""




