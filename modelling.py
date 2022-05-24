from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import joblib

# get data
iris = datasets.load_iris()

# data prep
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# modelling
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

# save model
joblib.dump(model, "app/model.joblib")
