from pyspark import SparkContext, SparkConf
import os
os.environ["SPARK_HOME"] = "/Users/ivanmartin/Software/spark-1.6.1"
conf = (SparkConf().setMaster('local').setAppName('a'))
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer, PunktSentenceTokenizer

import re
from nltk.metrics import *

sc = SparkContext()

#input = [1, 0.0, "IE", "He met U.S. President, George W. Bush, in Washington and British Prime Minister, Tony Blair, in London.", "Washington is a part of London."]
def extract_features_and_target(input):

    #cachedStopWords = stopwords.words("english") #common english words
    #lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    t = input[3]
    h = input[4]
    entail = input[1]
    output = []

    def tokenize(text):
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text))
        tokens =(list(map(lambda token: PorterStemmer().stem(token), words)))
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
        return filtered_tokens

    # Strings as Tokens
    t_token = tokenize(input[3])
    h_token = tokenize(input[4])
    # Token as sets
    t_set = set(t_token)
    h_set = set(h_token)

    def count_equal(hypo):
        x0 = 0
        for word in h_set:
            if word in t_set:
                x0 += 1
        return x0

    # number of appearing words on both T and H
    x0 = count_equal(h_token)

    # Metrics on sets
    x2 = precision(t_set, h_set)
    x3 = recall(t_set, h_set)
    x4 = f_measure(t_set, h_set)

    # Different distances
    x5 = edit_distance(t, h) # Levensgtein on string
    x6 = edit_distance(t_token, h_token) # Levensgtein on tokenized
    x7 = binary_distance(t_set, h_set) # !!! Seems to always throw 1.0
    x8 = jaccard_distance(t_set, h_set) # Jaccard on sets
    x9 = masi_distance(t_set, h_set) # Masi on sets

    output = [x0, x2, x3, x4, x5, x6, x7, x8, x9, entail]

    return LabeledPoint(output[-1], output[:-1])

textFile = sc.textFile("/Users/ivanmartin/Google Drive/IE BD Master/NLP/TextEntilmentDevelopment/NLTK_sentence_entailments/training_set.xml")

original_data = textFile.map(lambda x:[x.split("id=\"",1)[1].split("\"",1)[0],
                                   x.split("entailment=\"",1)[1].split("\"",1)[0],
                                   x.split("task=\"",1)[1].split("\"",1)[0],
                                   x.split("<t>",1)[1].split("</t>",1)[0],
                                   x.split("<h>",1)[1].split("</h>",1)[0]])

original_data_ready_to_extract_features = original_data.map(lambda x: [int(x[0]), 1.0, x[2], x[3], x[4]] if (x[1] == "YES") else [int(x[0]), 0.0, x[2], x[3], x[4]])

training_features_with_target = original_data_ready_to_extract_features.map(lambda x: extract_features_and_target(x))
#training_features_with_target = original_data_ready_to_extract_features.map(extract_features_and_target)


# Build the model
model = SVMWithSGD.train(training_features_with_target, iterations=100)

# Evaluating the model on training data
labelsAndPreds = training_features_with_target.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda realAndPrediction: realAndPrediction[0] != realAndPrediction[1]).count() / float(training_features_with_target.count())
print("Training Error = " + str(trainErr))

# Save and load model
#model.save(sc, "myModelPath")
#sameModel = SVMModel.load(sc, "myModelPath")
