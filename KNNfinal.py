import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay 


malignant = 1
benign = -1

inputData = pd.read_csv("wdbc.data.mb (1).csv")
print("Main statistics of the dataset:")
inputData.describe()
print("\nMean of the dataset:")
print(inputData.mean)
print("\nFirst few rows of the data:")
print(inputData.head())

inputArray = inputData.to_numpy()
X = inputArray[:, :-1]
y = inputArray[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

class KNN:
    def __init__(self, k):
        self.k = k
        self.training_data = None
        self.training_labels = None

    def fit(self, X, y):
        self.training_data = X
        self.training_labels = y

    def predict(self, X_test):
        predictions = []
        for test_point in X_test:
            distances = []
            for i, training_point in enumerate(self.training_data):
                distance = euclidean_distance(test_point, training_point)
                distances.append((distance, i))
            distances.sort(key=lambda x: x[0])
            k_nearest_neighbors = distances[:self.k]
            neighbor_labels = [self.training_labels[i] for _, i in k_nearest_neighbors]
            prediction = max(set(neighbor_labels), key=neighbor_labels.count)
            predictions.append(prediction)
        return predictions 


knn = KNN(k=9)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.4f}")

class_names = [malignant, benign]
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]


for titles, normalize in titles_options:
    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,  # Pass true labels (y_test) and predictions
        display_labels=class_names,
        cmap=plt.cm.inferno,
        normalize=normalize,
    )
    display.ax_.set_title(titles)
    print(titles)
    print(display.confusion_matrix)

plt.show()
'''for titles, normalize in titles_options:
    display = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions
        display_labels=class_names,
        cmap=plt.cm.inferno,
        normalize=normalize,
    )
    display.ax_.set_title(title)
    print(title)
    print(display.confusion_matrix)'''

plt.show()