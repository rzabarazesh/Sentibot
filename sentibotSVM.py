__author__ = 'mohammadreza'





__author__ = 'mohammadreza'
import re
import csv
import pickle
import svm
from svmutil import *

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



AllFeatures = list(set(AllFeatures))   # feture list

###




### START OF SVM TRAINING


def getSVMFeatureVectorAndLabels(tweets, featureList):
    sortedFeatures = sorted(featureList)
    map = {}
    feature_vector = []
    labels = []
    for t in tweets:
        label = 0
        map = {}
        #Initialize empty map
        for w in sortedFeatures:
            map[w] = 0

        tweet_words = t[0]
        tweet_opinion = t[1]
        #Fill the map
        for word in tweet_words:
            #process the word (remove repetitions and punctuations)
            word = replaceTwoOrMore(word)
            word = word.strip('\'"?,.')
            #set map[word] to 1 if word exists
            if word in map:
                map[word] = 1
        #end for loop
        values = map.values()
        feature_vector.append(values)
        if(tweet_opinion == 'positive'):
            label = 0
        elif(tweet_opinion == 'negative'):
            label = 1
        elif(tweet_opinion == 'neutral'):
            label = 2
        labels.append(label)
    #return the list of feature_vector and labels
    return {'feature_vector' : feature_vector, 'labels': labels}
#end

#Train the SVM
result = getSVMFeatureVectorandLabels(tweets, featureList)
problem = svm_problem(result['labels'], result['feature_vector'])
#'-q' DONT SIDPLAY CONSOLE OUTPUT
param = svm_parameter('-q')
param.kernel_type = LINEAR
classifier = svm_train(problem, param)
svm_save_model(classifierDumpFile, classifier)

#Test the classifier
test_feature_vector = getSVMFeatureVector(test_tweets, featureList)
#p_labels contains the final labeling result
p_labels, p_accs, p_vals = svm_predict([0] * len(test_feature_vector),test_feature_vector, classifier)