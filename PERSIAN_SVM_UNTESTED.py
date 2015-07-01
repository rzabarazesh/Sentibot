__author__ = 'mohammadreza'
__author__ = 'mohammadreza'

# AN AI PROJECT
# PLZ INSTALL PYTHON 2.6 AND SVMLIB + BINDINGS

import re
import csv
import pickle
import svm
from svmutil import *

_stopwordFileName = "stopwords.txt"

def preProcess(tweet):

    #remove links
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    #remove @'s
    tweet = re.sub('@[^\s]+','',tweet)  ## aya \S farsi ham hast ? test shavad ????????
    #remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)  ## EYZAN .................
    #Replace #tag with tag
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return (tweet)

def simplifyLongWords(w):
    #simplify words with more than 2 recurring characters "salaaaaaaam" ,"salaaaam" , "salaaaaaaaaaam" ===> "salaam"
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return (pattern.sub(r"\1\1", w))

def getStopWords():
    stopwords = []
    for line in open(_stopwordFileName).readlines():
        stopwords.append(line.strip("\n"))
    return (stopwords)
stopwords = getStopWords()

def getFeatures(tweet):
    v = [] #feature vector
    bagOfWords = tweet.split()
    for word in bagOfWords :
        word = simplifyLongWords(word)
        word = word.strip('\'"?.,.')

        if(word in stopwords ):
            continue
        else :
            v.append(word)
    return v



#### START OF PROGRAM
print("......INITIATING THE SENTI BOT ....")
print("- CODED BY MOHAMMAD REZA BARAZESH -")
print("___________ AL73rnA _______________")
####
learningSet = csv.reader(open("learning.csv"))
AllFeatures = []
AllTweets = []

def featureVector(tweet):
    tweet_words = set(tweet)
    features = {}
    for word in AllFeatures:
        features[word] = (word in tweet_words)
    return features


AllFeatures = list(set(AllFeatures))   # feture list
###
print("-Training a new classifier ...")
print("-Reading Learning set file")
###
for line in learningSet :
    sentiment = line[0]
    tweet = line[1]
    processedTweet = preProcess(tweet)
    tweetFeatures = getFeatures(processedTweet)
    AllFeatures.extend(tweetFeatures)
    AllTweets.append((tweetFeatures,sentiment))
###

### REST IS THE SAME AS SVM !!!!!!!!!! JUST COPY