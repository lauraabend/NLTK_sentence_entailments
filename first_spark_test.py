from pyspark import SparkContext
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext()

####################################################################################################
################################# The code from Ivan should also include the label##################
####################################################################################################
# Load and parse the data
#def parsePoint(line):
#    values = [float(x) for x in line.split(' ')]
#    return LabeledPoint(values[0], values[1:])
####################################################################################################
####################################################################################################
def extract_features_and_target(X):
    # TODO: Assume target at the end here
    return LabeledPoint(X[-1], X[:-2])

textFile = sc.textFile("/Users/ivanmartin/Google Drive/IE BD Master/NLP/TextEntilmentDevelopment/training_set.xml")

original_data = textFile.map(lambda x:[x.split("id=\"",1)[1].split("\"",1)[0],
                                   x.split("entailment=\"",1)[1].split("\"",1)[0],
                                   x.split("task=\"",1)[1].split("\"",1)[0],
                                   x.split("<t>",1)[1].split("</t>",1)[0],
                                   x.split("<h>",1)[1].split("</h>",1)[0]])

original_data_ready_to_extract_features = original_data.map(lambda x: [int(x[0]), True, x[2], x[3], x[4]] if (x[1] == "YES") else [int(x[0]), False, x[2], x[3], x[4]])

training_features_with_target = original_data_ready_to_extract_features.map(lambda x: extract_features_and_target(x))

training_features_with_target.take(1)

# Build the model
# TODO: REMEMBER TO UPDATE THE NUMBER OF COLUMNS
model = SVMWithSGD.train(training_features_with_target[2:5], iterations=100)

# Evaluating the model on training data
labelsAndPreds = training_features_with_target.map(lambda p: (training_features_with_target[1], model.predict(training_features_with_target[2:5])))
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(training_features_with_target.count())
print("Training Error = " + str(trainErr))

# Save and load model
model.save(sc, "myModelPath")
sameModel = SVMModel.load(sc, "myModelPath")