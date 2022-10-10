import pandas as pd
import snscrape.modules.twitter as sntwitter
import numeritos as nitos
pd.set_option('display.max_colwidth',1000)
from scipy.special import softmax
import pandas as pd
import xml.etree.ElementTree as ET
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from collections import Counter
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import tensorflow as tf
from datetime import datetime
import sqlite3
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
import os
import pickle
from nltk.corpus import stopwords 



#Hacemos una consulta 
query='thebridge_tech (@thebridge_tech) until:2022-10-05 since:2022-06-13'
tweets= []
limit=100000


for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets)== limit:
        break
    else:
        tweets.append([tweet.date,tweet.user.id,tweet.user.displayname,tweet.user.username, tweet.content,tweet.conversationId,tweet.retweetCount,tweet.replyCount, tweet.likeCount,tweet.mentionedUsers.count,tweet.user.followersCount, tweet.quoteCount])

df=pd.DataFrame(tweets, columns=['Date','User_id','User','Username','Content','Content_id','Retweet_count','Reply','Likes','Mentions','Followers','Quote' ])

#2 Crear un SQL
author_df=df[['User','Username']]
tweet_df=df[['Content','Content_id','Retweet_count','Reply','Likes','Followers','Quote' ]]

connection = sqlite3.connect('data/twitter.db')
tweet_df.to_sql('tweets', con=connection, index=False)
author_df.to_sql('users', con=connection, index=False)
connection.close()
