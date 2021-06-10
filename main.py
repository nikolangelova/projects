
# Written by Nikol Angelova

# This is what I used to scrape the comments :
# https://github.com/egbertbouman/youtube-comment-downloader

# this line was run in terminal
# youtube-comment-downloader --youtubeid 24Z9l5yZtkg --output 24Z9l5yZtkg.json


import json
import pandas as pd
import string
import re
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
nltk.download('punkt')
nltk.download('stopwords')

with open('/Users/nikolangelova/PycharmProjects/24Z9l5yZtkg.json') as f:
  data = json.load(f)

print(data)

df = pd.DataFrame(data)
df.head()

# ======================================================================
# Feature Creation before cleaning of the data
# ======================================================================

# First feature we can create is the number of stopwords in each comment
stop = stopwords.words('english')
df['stopwords'] = df['text'].apply(lambda x: len([x for x in x.split() if x in stop]))
df[['text','stopwords']].head()
#                                                 text  stopwords
# 0                      Where is the kitty litter ???          2
# 1  This must be a place for someone who doesn’t e...         14
# 2  The place is stunning! I’m curious to know how...          5
# 3  Damn, that counter top is incredible! 💦 And th...         4
# 4  The PANTRY / CLOSET should be all WRAPPED AROU...          5


# Number of punctuation - this can be an indicator of an emotionally charged comment
def count_punct(text):
  count = sum([1 for char in text if char in string.punctuation])
  return count

df['punctuation'] = df['text'].apply(lambda x: count_punct(x))

df[['text','punctuation']].head()
#                                                 text  punctuation
# 0                      Where is the kitty litter ???            3
# 1  This must be a place for someone who doesn’t e...            3
# 2  The place is stunning! I’m curious to know how...            1
# 3  Damn, that counter top is incredible! 💦 And th...           2
# 4  The PANTRY / CLOSET should be all WRAPPED AROU...            3

# Number of numerics present in the comments

df['numerics'] = df['text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df[['text','numerics']].head()
#                                                 text  numerics
# 0                      Where is the kitty litter ???         0
# 1  This must be a place for someone who doesn’t e...         0
# 2  The place is stunning! I’m curious to know how...         0
# 3  Damn, that counter top is incredible! 💦 And th...        0
# 4  The PANTRY / CLOSET should be all WRAPPED AROU...         0

# Uppercase words - uppercase words can be an indicator of rage in comments ,
# or also very strong positive emotions

df['upper'] = df['text'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
df[['text','upper']].head()
#                                                 text  upper
# 0                      Where is the kitty litter ???      0
# 1  This must be a place for someone who doesn’t e...      2
# 2  The place is stunning! I’m curious to know how...      0
# 3  Damn, that counter top is incredible! 💦 And th...     0
# 4  The PANTRY / CLOSET should be all WRAPPED AROU...      8

# ==================================================================================
# Cleaning of the data
# ==================================================================================

# make all text lowercase
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df['text'].head()

# remove punctuation
df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'].head()

# remove stopwords
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df['text'].sample(10)


# remove emojis
# Reference : https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags 
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

# remove all emojis from df
df['text'] = df['text'].apply(lambda x: remove_emoji(x))

# correct spelling
df['text'] = df['text'].apply(lambda x: str(TextBlob(x).correct()))

# ======================================================================
# Feature Creation after cleaning of the data
# ======================================================================

# number of words
df['word_count'] = df['text'].apply(lambda x: len(str(x).split(" ")))
df[['text','word_count']].head()
#                                                 text  word_count
# 0                                       kitty litter           2
# 1  must place someone doesn ever use kitchen ll n...          14
# 2  place stunning in curious know expensive projects           7
# 3  damn counter top incredible timber defining be...           8
# 4  pantry closet wrapped around mirror face windo...           8

# number of characters
df['char_count'] = df['text'].str.len() ## this also includes spaces
df[['text','char_count']].head()
#                                                 text  char_count
# 0                                       kitty litter          12
# 1  must place someone doesn ever use kitchen ll n...          82
# 2  place stunning in curious know expensive projects          49
# 3  damn counter top incredible timber defining be...          56
# 4  pantry closet wrapped around mirror face windo...          54

# average word length
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/(len(words)+0.000001))

df['avg_word'] = df['text'].apply(lambda x: avg_word(x)).round(1)
df[['text','avg_word']].head()
#                                                 text  avg_word
# 0                                       kitty litter       5.5
# 1  must place someone doesn ever use kitchen ll n...       4.9
# 2  place stunning in curious know expensive projects       6.1
# 3  damn counter top incredible timber defining be...       6.1
# 4  pantry closet wrapped around mirror face windo...       5.9

# ======================================================================
# Sentiment Analysis of Comments using textblob
# ======================================================================
sentiment_scores_tb = [round(TextBlob(comment).sentiment.polarity, 3) for comment in df['text']]
sentiment_category_tb = ['positive' if score > 0
                             else 'negative' if score < 0
                                 else 'neutral'
                                     for score in sentiment_scores_tb]

df['sentiment_scores_tb'] = sentiment_scores_tb
df['sentiment_category_tb'] = sentiment_category_tb

# ======================================================================
# Vectorize comments in two ways
# ======================================================================

vect1 = CountVectorizer().fit(df.text)
vect2 = TfidfVectorizer().fit(df.text)

count_vect = vect1.transform(df.text)
tfidf_vect = vect2.transform(df.text)

count_vect_df = pd.DataFrame(count_vect.toarray(), columns = vect1.get_feature_names())
tfidf_vect_df = pd.DataFrame(tfidf_vect.toarray(), columns = vect2.get_feature_names())


# ======================================================================
# Count the top 50 most frequent words
# ======================================================================
freq = pd.Series(' '.join(df['text']).split()).value_counts()[:50]
freq
# love          259
# kitchen       247
# cat           247
# beautiful     243
# apartment     226
# design        199
# space         174
# bathroom      150
# one           148
# small         140
# like          125
# bed           111
# amazing       109
# would         109
# bedroom       104
# now           103
# great          85
# really         79
# well           79
# give           74
# mirror         74
# also           72
# door           70
# looks          66
# seen           66
# stunning       66
# video          65
# absolutely     65
# best           65
# gorgeous       63
# cook           61
# nice           61
# much           61
# place          59
# living         59
# see            54
# table          53
# done           52
# in             51
# work           51
# white          49
# area           49
# hidden         48
# never          47
# live           47
# even           46
# counter        45
# use            45
# architect      44
# look           44

# ======================================================================
# Create wordcloud
# ======================================================================

text = " ".join(comment for comment in df.text)

wordcloud = WordCloud().generate(text)

# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

plt.savefig("/Users/nikolangelova/PycharmProjects/Wordcloud.png", format="png")
