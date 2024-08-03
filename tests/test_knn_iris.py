import unittest
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
 
class TestKNNIris(unittest.TestCase):
 
    def setUp(self):
        # Load Iris dataset
        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        self.knn = KNeighborsClassifier(n_neighbors=3)
 
    def test_knn_accuracy(self):
        # Train the model
        self.knn.fit(self.X_train, self.y_train)
        # Predict with the model
        y_pred = self.knn.predict(self.X_test)
        # Calculate the accuracy of the model
        accuracy = accuracy_score(self.y_test, y_pred)
        self.assertGreaterEqual(accuracy, 0.9, "Accuracy should be at least 90")
 
if __name__ == '__main__':
    unittest.main()
