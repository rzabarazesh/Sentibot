__author__ = 'mohammadreza'
import nltk
import re
import csv
import pickle
_stopwordFileName = "stopwords.txt"

def preProcess(tweet):
    #to lower case
    tweet = tweet.lower()
    #remove links
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',tweet)
    #remove @'s
    tweet = re.sub('@[^\s]+','',tweet)
    #remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #tag with tag
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.strip('\'"')
    return (tweet)

def simplifyLongWords(w):
    #simplify words with more than 2 recurring characters "salaaaaaaam" ,"salaaaam" , "salaaaaaaaaaam" ===> "salaam"
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return (pattern.sub(r"\1\1", w))
def save_classifier(classifier):
   f = open("savedClassifier", 'wb')
   pickle.dump(classifier, f)
   f.close()

def load_classifier():
   f = open("savedClassifier", 'rb')
   classifier = pickle.load(f)
   f.close()
   return classifier

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
        startsWithAZ = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", word)
        if(word in stopwords or startsWithAZ == None):
            continue
        else :
            v.append(word.lower())
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



AllFeatures = list(set(AllFeatures))

###
print("-prepairing the NBC")
print("===================")
# Q=input("do u have a/ want to use a saved classifier ? y/n")
Q= "n"
if Q == "y" :
    NBClassifier = load_classifier()
if Q == "n" :
    ###
    print("-Training a new classifier ...")
    ###
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
    print("-Training the classifier")
    ###
    training_set = nltk.classify.util.apply_features(featureVector, AllTweets)
    NBClassifier = nltk.NaiveBayesClassifier.train(training_set)

    # Q2= input("do u want to save the new classifier ? y/n ")
    # if Q2 == "y" :
    #     save_classifier(NBClassifier)
    # else:
    #     pass

analysing = True
while(analysing):
    testTweet = input("enter tweet to analys ?")
    processedTestTweet = preProcess(testTweet)
    print (NBClassifier.classify(featureVector(getFeatures(processedTestTweet))))
    Q3=input("Continue ? y/n")
    if Q3 == "y":
        analysing = False

print(NBClassifier.show_most_informative_features())