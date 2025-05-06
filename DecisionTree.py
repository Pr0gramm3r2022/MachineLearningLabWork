import sklearn as sk
from sklearn import datasets
# import some data
iris = datasets.load_iris()
#retrieve the data
x = iris.data
y = iris.target
print(x)