from pyspark import SparkContext, SparkConf
import os
os.environ["SPARK_HOME"] = "/Users/ivanmartin/Software/spark-1.6.1"
conf = (SparkConf().setMaster('local').setAppName('a'))
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer, PunktSentenceTokenizer

import re, math
from nltk.metrics import *
from collections import Counter

sc = SparkContext()

#input = [1, 0.0, "IE", "He met U.S. President, George W. Bush, in Washington and British Prime Minister, Tony Blair, in London.", "Washington is a part of London."]
def extract_features_and_target(input):

    cachedStopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
    t = input[3]
    h = input[4]
    entail = input[1]
    output = []

    def tokenize(text):
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text));
        words = [word for word in words if word not in cachedStopWords]
        tokens =(list(map(lambda token: PorterStemmer().stem(token),
                      words)));
        p = re.compile('[a-zA-Z]+');
        filtered_tokens = list(filter(lambda token:
            p.match(token) and len(token)>=min_length, tokens));
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
    x1 = precision(t_set, h_set)
    x2 = recall(t_set, h_set)
    x3 = f_measure(t_set, h_set)

    # Different distances
    x4 = edit_distance(t, h) # Levensgtein on string
    x5 = edit_distance(t_token, h_token) # Levensgtein on tokenized
    x6 = binary_distance(t_set, h_set) # !!! Seems to always throw 1.0
    x7 = jaccard_distance(t_set, h_set) # Jaccard on sets
    x8 = masi_distance(t_set, h_set) # Masi on sets

    # Cosine distance
    WORD = re.compile(r'\w+')
    # Mathematical expression
    def get_cosine(vec1, vec2):
         intersection = set(vec1.keys()) & set(vec2.keys())
         numerator = sum([vec1[x] * vec2[x] for x in intersection])

         sum1 = sum([vec1[x]**2 for x in vec1.keys()])
         sum2 = sum([vec2[x]**2 for x in vec2.keys()])
         denominator = math.sqrt(sum1) * math.sqrt(sum2)

         if not denominator:
            return 0.0
         else:
            return float(numerator) / denominator
    # T and H strings to vectors
    def text_to_vector(text):
         words = WORD.findall(text)
         return Counter(words)

    vector1 = text_to_vector(t)
    vector2 = text_to_vector(h)

    x9 = get_cosine(vector1, vector2)

    output = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, entail]


    return LabeledPoint(output[-1], output[:-1])

textFile = sc.textFile("/Users/ivanmartin/Google Drive/IE BD Master/NLP/TextEntilmentDevelopment/NLTK_sentence_entailments/training_set.xml")

original_data = textFile.map(lambda x:[x.split("id=\"",1)[1].split("\"",1)[0],
                                   x.split("entailment=\"",1)[1].split("\"",1)[0],
                                   x.split("task=\"",1)[1].split("\"",1)[0],
                                   x.split("<t>",1)[1].split("</t>",1)[0],
                                   x.split("<h>",1)[1].split("</h>",1)[0]])

original_data_ready_to_extract_features = original_data.map(lambda x: [int(x[0]), True, x[2], x[3], x[4]] if (x[1] == "YES") else [int(x[0]), False, x[2], x[3], x[4]])

training_features_with_target = original_data_ready_to_extract_features.map(lambda x: extract_features_and_target(x))
#training_features_with_target = original_data_ready_to_extract_features.map(extract_features_and_target)

results = []
#for i in [5,10,15]:
for i in [300]:
    # Build the model
    model = SVMWithSGD.train(training_features_with_target, iterations=i)

# Evaluating the model on training data
    labelsAndPreds = training_features_with_target.map(lambda p: (p.label, not model.predict(p.features)))
    trainErr = labelsAndPreds.filter(lambda realAndPrediction: realAndPrediction[0] != realAndPrediction[1]).count() / float(training_features_with_target.count())
    print("Training Error = " + str(trainErr))
    results.append(trainErr)

import matplotlib.pyplot as plt
plt.plot(results)
plt.show()

# Save and load model
#model.save(sc, "myModelPath")
#sameModel = SVMModel.load(sc, "myModelPath")
