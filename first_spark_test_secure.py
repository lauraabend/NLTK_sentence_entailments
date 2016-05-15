from __future__ import division
from pyspark import SparkContext, SparkConf
import os
os.environ["SPARK_HOME"] = "/Users/ivanmartin/Software/spark-1.6.1"
conf = (SparkConf().setMaster('local').setAppName('a'))


from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import GradientBoostedTrees
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer, PunktSentenceTokenizer

import re, math
from nltk.metrics import *
from collections import Counter

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
import math
import numpy as np
import sys

ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0

sc = SparkContext()

def get_best_synset_pair(word_1, word_2):
    """
    Choose the pair with highest path similarity among all pairs.
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
               sim = wn.path_similarity(synset_1, synset_2)
               print(str(synset_1) + " " + str(synset_2) + " " + str(sim))
               if (sim != None) and (sim > max_sim):
                   max_sim = sim
                   best_pair = synset_1, synset_2
        return best_pair


def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic
    ontology (Wordnet in our case as well as the paper's) between two
    synsets.
    """
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)


def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]:x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]:x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if lcs_candidate in hypernyms_1.keys():
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if lcs_candidate in hypernyms_2.keys():
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) /
        (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))

def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return (length_dist(synset_pair[0], synset_pair[1]) *
        hierarchy_dist(synset_pair[0], synset_pair[1]))

######################### sentence similarity ##########################

def most_similar_word(word, word_set):
    """
    Find the word in the joint word set that is most similar to the word
    passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar
    word and the actual similarity value.
    """
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
      sim = word_similarity(word, ref_word)
      if (sim != None) and (sim > max_sim):
          max_sim = sim
          sim_word = ref_word
    return sim_word, max_sim

def info_content(lookup_word):
    """
    Uses the Brown corpus available in NLTK to calculate a Laplace
    smoothed frequency distribution of words, then uses this information
    to compute the information content of the lookup_word.
    """
    global N
    if N == 0:
        # poor man's lazy evaluation
        for sent in brown.sents():
            for word in sent:
                word = word.lower()
                if word not in brown_freqs:
                    brown_freqs[word] = 0
                brown_freqs[word] = brown_freqs[word] + 1
                N = N + 1
    lookup_word = lookup_word.lower()
    n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]
    return 1.0 - (math.log(n + 1) / math.log(N + 1))

def semantic_vector(words, joint_words, info_content_norm):
    """
    Computes the semantic vector of a sentence. The sentence is passed in as
    a collection of words. The size of the semantic vector is the same as the
    size of the joint word set. The elements are 1 if a word in the sentence
    already exists in the joint word set, or the similarity of the word to the
    most similar word in the joint word set if it doesn't. Both values are
    further normalized by the word's (and similar word's) information content
    if info_content_norm is True.
    """
    sent_set = set(words)
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            if info_content_norm:
                semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = PHI if max_sim > PHI else 0.0
            if info_content_norm:
                semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1
    return semvec

def semantic_similarity(sentence_1, sentence_2, info_content_norm):
    """
    Computes the semantic similarity between two sentences as the cosine
    similarity between the semantic vectors computed for each sentence.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    vec_1 = semantic_vector(words_1, joint_words, info_content_norm)
    vec_2 = semantic_vector(words_2, joint_words, info_content_norm)
    return np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))

######################### word order similarity ##########################

def word_order_vector(words, joint_words, windex):
    """
    Computes the word order vector for a sentence. The sentence is passed
    in as a collection of words. The size of the word order vector is the
    same as the size of the joint word set. The elements of the word order
    vector are the position mapping (from the windex dictionary) of the
    word in the joint set if the word exists in the sentence. If the word
    does not exist in the sentence, then the value of the element is the
    position of the most similar word in the sentence as long as the similarity
    is above the threshold ETA.
    """
    wovec = np.zeros(len(joint_words))
    i = 0
    wordset = set(words)
    for joint_word in joint_words:
        if joint_word in wordset:
            # word in joint_words found in sentence, just populate the index
            wovec[i] = windex[joint_word]
        else:
            # word not in joint_words, find most similar word and populate
            # word_vector with the thresholded similarity
            sim_word, max_sim = most_similar_word(joint_word, wordset)
            if max_sim > ETA:
                wovec[i] = windex[sim_word]
            else:
                wovec[i] = 0
        i = i + 1
    return wovec

def word_order_similarity(sentence_1, sentence_2):
    """
    Computes the word-order similarity between two sentences as the normalized
    difference of word order between the two sentences.
    """
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = list(set(words_1).union(set(words_2)))
    windex = {x[1]: x[0] for x in enumerate(joint_words)}
    r1 = word_order_vector(words_1, joint_words, windex)
    r2 = word_order_vector(words_2, joint_words, windex)
    return 1.0 - (np.linalg.norm(r1 - r2) / np.linalg.norm(r1 + r2))

######################### overall similarity ##########################

def similarity(sentence_1, sentence_2, info_content_norm):
    """
    Calculate the semantic similarity between two sentences. The last
    parameter is True or False depending on whether information content
    normalization is desired or not.
    """
    return DELTA * semantic_similarity(sentence_1, sentence_2, info_content_norm) + \
        (1.0 - DELTA) * word_order_similarity(sentence_1, sentence_2)

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

    #x10 = similarity(t, h, True)
    #output = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, entail]
    output = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, entail]

    return LabeledPoint(output[-1], output[:-1])

textFile = sc.textFile("/Users/ivanmartin/Google Drive/IE BD Master/NLP/TextEntilmentDevelopment/NLTK_sentence_entailments/training_set.xml")
textFile_test = sc.textFile("/Users/ivanmartin/Google Drive/IE BD Master/NLP/TextEntilmentDevelopment/NLTK_sentence_entailments/textual_entailment_test.xml")

original_data = textFile.map(lambda x:[x.split("id=\"",1)[1].split("\"",1)[0],
                                   x.split("entailment=\"",1)[1].split("\"",1)[0],
                                   x.split("task=\"",1)[1].split("\"",1)[0],
                                   x.split("<t>",1)[1].split("</t>",1)[0],
                                   x.split("<h>",1)[1].split("</h>",1)[0]])

original_data_test = textFile_test.map(lambda x:[[x.split("id=\"",1)[1].split("\"",1)[0],
                                                 "YES",
                                                 x.split("task=\"",1)[1].split("\"",1)[0],
                                                 x.split("<t>",1)[1].split("</t>",1)[0],
                                                 x.split("<h>",1)[1].split("</h>",1)[0]],x])

original_data_ready_to_extract_features = original_data.map(lambda x: [int(x[0]), True, x[2], x[3], x[4]] if (x[1] == "YES") else [int(x[0]), False, x[2], x[3], x[4]])
original_data_ready_to_extract_features_test = original_data_test.map(lambda x: [[int(x[0][0]), True, x[0][2], x[0][3], x[0][4]],x[1]] if (x[0][1] == "YES") else [[int(x[0][0]), False, x[0][2], x[0][3], x[0][4]],x[1]])


training_features_with_target = original_data_ready_to_extract_features.map(lambda x: extract_features_and_target(x))
training_features_with_target_test = original_data_ready_to_extract_features_test.map(lambda x: (x[0][0],x[1], extract_features_and_target(x[0])))
#training_features_with_target = original_data_ready_to_extract_features.map(extract_features_and_target)

results = []


from pyspark.mllib.tree import RandomForest

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = training_features_with_target.randomSplit([0.7, 0.3])

# Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose.
print("Training model with 70% of the data")
model = RandomForest.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=4, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=3, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
#predictions_Adjusted = predictions.map(lambda x: not x)
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
#testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
testErr = labelsAndPredictions.filter(lambda x: x[0] != x[1]).count() / float(testData.count())

print('Test Error with 30% of the data= ' + str(testErr))

print("Now training with 100% of the data but same configuration as in the previous test")
model = RandomForest.trainClassifier(training_features_with_target, numClasses=2, categoricalFeaturesInfo={},
                                     numTrees=4, featureSubsetStrategy="auto",
                                     impurity='gini', maxDepth=3, maxBins=32)

print("Using model with 100% training data to predict validation set")
#Predictions_of_test_set = training_features_with_target_test.map(lambda p: (p[0],p[1], model.predict(p[2].features)))

# Evaluate model on test instances and compute test error
predictions = model.predict(training_features_with_target_test.map(lambda x: x[2].features))
Predictions_of_test_set_with_subarray = training_features_with_target_test.map(lambda p: [p[0], p[1]]).zip(predictions)
Predictions_of_test_set = Predictions_of_test_set_with_subarray.map(lambda x:[x[0][0], x[0][1],x[1]])

Test_data_with_prediction = Predictions_of_test_set.map(lambda p: p[1].split("task=")[0] + str("entailment=\"YES\" " if p[2]==True else "entailment=\"NO\" ") + "task=" +p[1].split("task=")[1])

print("Saving predictions")
Test_data_with_prediction.saveAsTextFile("/Users/ivanmartin/Google Drive/IE BD Master/NLP/TextEntilmentDevelopment/NLTK_sentence_entailments/textual_entailment_test_with_prediction.xml")




'''
# Build the model
model = SVMWithSGD.train(training_features_with_target, iterations=300)
#model = RandomForest.trainClassifier(training_features_with_target, numClasses=2, categoricalFeaturesInfo={},
#                                 numTrees=30, featureSubsetStrategy="auto", impurity='gini', maxDepth=8, maxBins=32)
#model = GradientBoostedTrees.trainClassifier(training_features_with_target, categoricalFeaturesInfo={}, numIterations=3)
# Evaluating the model on training data
predictions = model.predict(training_features_with_target.map(lambda x: x.features))
labelsAndPredictions = training_features_with_target.map(lambda lp: lp.label).zip(predictions)
#labelsAndPreds = training_features_with_target.map(lambda p: (p.label, not model.predict(p.features)))
for j in labelsAndPredictions.take(4):
    print(j)
#trainErr = labelsAndPreds.filter(lambda realAndPrediction: realAndPrediction[0] != realAndPrediction[1]).count() / float(training_features_with_target.count())
correctCount = labelsAndPredictions.filter(lambda realAndPrediction: realAndPrediction[0] != realAndPrediction[1]).count()
totalCount = float(training_features_with_target.count())
print("Training Error = " + str(correctCount/totalCount))
results.append(correctCount/totalCount)

Predictions_of_test_set = training_features_with_target_test.map(lambda p: (p[0],p[1], not model.predict(p[2].features)))

Test_data_with_prediction = Predictions_of_test_set.map(lambda p: p[1].split("task=")[0] + str("entailment=\"YES\" " if p[2]==True else "entailment=\"NO\" ") + "task=" +p[1].split("task=")[1])

for i in Test_data_with_prediction.take(5):
    print(i)

Test_data_with_prediction.saveAsTextFile("/Users/ivanmartin/Google Drive/IE BD Master/NLP/TextEntilmentDevelopment/NLTK_sentence_entailments/textual_entailment_test_with_prediction.xml")


'''