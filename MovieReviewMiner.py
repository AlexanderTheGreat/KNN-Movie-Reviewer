import KNN
import re
import string
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import datetime

# pre: receives one specific review, after being run through Beautiful Soup to remove xml tags
# post: returns formatted text with stopwords, and punctuation removed
def formatter(data):
    review = re.sub("[^a-zA-Z]", " ", BeautifulSoup(data, "lxml").get_text())  # removes non-alphabet characters and HTML tags
    cleanWords = [x for x in review.split() if x not in stopWords]  # clean up stopwords from list of split words

    return cleanWords


print("Initializing...", end="\r")
start = time.time()  # Timing how long the program runs
formattedTrainLines = []  # formatted and cut down list of all the training reviews
formattedTestLines = []  # formatted and cut down list of all the testing reviews
ratingsList = []  # List to hold separated + and - ratings from train.txt
stopWords = set(stopwords.words("english") + list(string.punctuation))  # words to ignore
print("Initializing [Complete]")

print("Reading Data...", end="\r")
# reading test and training data
trainData = open("train.txt", "r")
trainLines = trainData.readlines()  # itemizes each line in trainData, places the lines into the trainLines list
testData = open("test_file.txt", "r")
testLines = testData.readlines()  # itemizes each line in testData, places the lines into the testLines list
print("Reading Data [Complete]")

# Tokenizing words
print("Tokenizing Words...", end="\r")

# tokenizing testLines
for i in range(len(testLines)):
    print(f"Tokenizing Testing Data: {i + 1}/{len(testLines)}", end="\r")
    formattedTestLines.append(" ".join(map(str, formatter(testLines[i]))))  # adding formatted reviews to a formattedTesting list

for i in range(len(trainLines)):
    print(f"Tokenizing Training Data: {i + 1}/{len(trainLines)}", end="\r")
    ratingsList.append(trainLines[i][0])  # placing the + or - at the beginning of the review into ratingsList
    formattedTrainLines.append(" ".join(map(str, formatter(trainLines[i]))))  # adding formatted reviews to formattedTraining list

print("Tokenizing [Complete]")
print("Weighing Words...", end="\r")

# Weighing words
# use_idf="true" to work with cosine similarity, max_features = x limits the words we look at down to x amounts
# sublinear tf scaling used to prevent weighing a very common word too highly
tfidf = TfidfVectorizer(use_idf=True, sublinear_tf=True, max_features=10000)
trainingVector = tfidf.fit_transform(formattedTrainLines).toarray()
testingVector = tfidf.transform(formattedTestLines).toarray()


print("Weighing Words [Complete]")
print(f"Train Size: {trainingVector.shape}")
print(f"Test Size: {testingVector.shape}")

knn = KNN.KNN(testingVector, trainingVector, 155)
predictedScores = knn.predict(testingVector, trainingVector, ratingsList)

print("Writing Output...", end="\r")

with open("format.txt", "w") as format:
    for score in predictedScores:
        format.write(score + "\n")

print("Finished")
print(f"Time to complete: {str(datetime.timedelta(seconds=(time.time() - start)))}")