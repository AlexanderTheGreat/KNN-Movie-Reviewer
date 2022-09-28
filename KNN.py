import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt


class KNN(object):

    def __init__(self, testingVector, trainingVector, k):
        self.k = k
        self.testingVector, self.trainingVector = testingVector, trainingVector  # inputs, labels

    def getDistance(self):
        return distance.euclidean(self.testingVector, self.trainingVector, 1)  # input arrays and weight (default: 1)

    def predict(self, testingVector, trainingVector, ratingsList):
        predictedScores = []  # final predicted review scores, that will be output to format.txt
        status = 0  # amount of reviews completed

        # performing SVD
        print("Performing SVD transformation...", end="\r")
        svd = TruncatedSVD(n_components=3000)

        # transforming data
        newTrainingVector = svd.fit_transform(trainingVector)
        newTestingVector = svd.transform(testingVector)

        print("Performing SVD transformation [Completed]")
        print(f"New Train Size: {newTrainingVector.shape}")
        print(f"New Test Size: {newTestingVector.shape}")

        # charting
        plt.style.use('dark_background')
        x, y = newTrainingVector[:][0], newTrainingVector[:][1]
        x2, y2 = newTestingVector[:][0], newTestingVector[:][1]
        plt.scatter(x2, y2, color="green", marker='o', label='Testing')
        plt.scatter(x, y, color="red", marker='x', label='Training')
        plt.legend(loc='upper left')
        plt.show()

        print("Getting vector similarity...", end="\r")
        similarity = cosine_similarity(newTestingVector, newTrainingVector)
        print("Getting vector similarity [Completed]")

        print("Predicting...", end="\r")
        for i in similarity:
            badReviews = 0
            goodReviews = 0
            neighbors = np.argpartition(-i, self.k)[0:self.k]   # sort and get the k neighbors, partition into k

            for i in neighbors:
                if ratingsList[i] == "-":
                    badReviews += 1
                else:
                    goodReviews += 1

            if badReviews > goodReviews:  # no ties, since K must be odd
                result = "-1"
            else:
                result = "+1"

            status += 1
            print(f"Predicted Review {status}/{len(similarity)}", end="\r")
            predictedScores.append(result)

        print("Predicting [Completed]")

        return predictedScores
