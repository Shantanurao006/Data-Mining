# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:06:55 2017

@author: Kalu
"""

import twitter
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import Birch
from sklearn.manifold import TSNE
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_duplicates(values):
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output

def left(s, amount = 1, substring = ""):
    if (substring == ""):
        return s[:amount]
    else:
        if (len(substring) > amount):
            substring = substring[:amount]
        return substring + s[:-amount]

def right(s, amount = 1, substring = ""):
    if (substring == ""):
        return s[-amount:]
    else:
        if (len(substring) > amount):
            substring = substring[:amount]
        return s[:-amount] + substring


CONSUMER_KEY ="f8kaJWDHuYV0TtAXpjZpsjoM7"
CONSUMER_SECRET ="ST8q2Gy23MB5mMe9MwJgVnayIVDF7xTmRgwlpB2mwXjst5Ljg6"
OAUTH_TOKEN = "834265704438304768-2dYeJWWBTqPImzZfKhtjtoDC0Czxp3m"
OAUTH_TOKEN_SECRET = "kJH0SZdESlMCyHiTp7711BgEQaLK0fBstiHaA68EFJlms"


auth = twitter.oauth.OAuth (OAUTH_TOKEN, OAUTH_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

twitter_api= twitter.Twitter(auth=auth)
q='#obama'
count=100


search_results = twitter_api.search.tweets (q=q, count=count)

statuses=search_results['statuses']


for _ in range(5):
   print ("Length of statuses", len(statuses))
   try:
        next_results = search_results['search_metadata']['next_results']
   except KeyError:   
       break

   kwargs=dict( [kv.split('=') for kv in next_results[1:].split("&") ])

   search_results = twitter_api.search.tweets(**kwargs)
   statuses += search_results['statuses']


print (json.dumps(statuses[0], indent=10))




hashtags = [ hashtag['text'].lower()
    for status in statuses
       for hashtag in status['entities']['hashtags'] ]


urls = [ urls['url']
    for status in statuses
       for urls in status['entities']['urls'] ]


texts = [ status['text'].lower()
    for status in statuses
        ]

created_ats = [ status['created_at']
    for status in statuses
        ]

# Preparing data for trending in the format: date word
i=0
print ("===============================\n")
for x in created_ats:
     for w in texts[i].split(" "):
        if len(w)>=2:
              print (x[4:10], x[26:31] ," ", w)
     i=i+1

# Prepare tweets data for clustering
# Converting text data into bag of words model

vectorizer = TfidfVectorizer(analyzer = "word", \
                             tokenizer = None,  \
                             preprocessor = None,  \
                             stop_words='english', \
                             max_features = 5000) 



for counter, t in enumerate(texts):
    if t.startswith("rt @"):
          pos= t.find(": ")
          texts[counter] = right(t, len(t) - (pos+2))
          
for counter, t in enumerate(texts):
    texts[counter] = re.sub(r'[?|$|.|!|#|\-|"|\n|,|@|(|)]',r'',texts[counter])
    texts[counter] = re.sub(r'https?:\/\/.*[\r\n]*', '', texts[counter], flags=re.MULTILINE)
    texts[counter] = re.sub(r'[0|1|2|3|4|5|6|7|8|9|:]',r'',texts[counter]) 
    texts[counter] = re.sub(r'deeplearning',r'deep learning',texts[counter])      
        
texts= remove_duplicates(texts)  

train_data_features = vectorizer.fit_transform(texts)
train_data_features = train_data_features.toarray()

print (train_data_features.shape)
print (train_data_features)

vocab = vectorizer.get_feature_names()
print (vocab)

dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print (count, tag)


# Clustering data
n_clusters=200
brc = Birch(branching_factor=50, n_clusters=n_clusters, threshold=0.239,  compute_labels=True)
brc.fit(train_data_features)

clustering_result=brc.predict(train_data_features)
print("==================================================================================")
print ("Clustering_result:\n")
print (clustering_result)

# Outputting some data
print (json.dumps(hashtags[0:50], indent=1))
print (json.dumps(urls[0:50], indent=1))
print (json.dumps(texts[0:50], indent=1))
print (json.dumps(created_ats[0:50], indent=1))


with open("d:\\tweet_data.txt", "a") as myfile:
     for w in hashtags: 
           myfile.write(str(w.encode('ascii', 'ignore')))
           myfile.write("\n")



# count of word frequencies
wordcounts = {}
for term in hashtags:
    wordcounts[term] = wordcounts.get(term, 0) + 1


items = [(v, k) for k, v in wordcounts.items()]
print (len(items))

xnum=[i for i in range(len(items))]
for count, word in sorted(items, reverse=True):
    print("%5d %s" % (count, word))
   


for x in created_ats:
  print (x)
  print (x[4:10])
  print (x[26:31])
  print (x[4:7])



plt.figure(1)
plt.title("Frequency of Hashtags")

myarray = np.array(sorted(items, reverse=True))

plt.xticks(xnum, myarray[:,1],rotation='vertical')
plt.plot (xnum, myarray[:,0])
plt.show()


model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
Y=model.fit_transform(train_data_features)
print (Y)


plt.figure(2)
plt.scatter(Y[:, 0], Y[:, 1], c=clustering_result, s=290,alpha=.5)

for j in range(len(texts)):    
   plt.annotate(clustering_result[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
   print ("%s %s" % (clustering_result[j],  texts[j]))
            
plt.show()