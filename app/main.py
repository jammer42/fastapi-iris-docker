import joblib
from fastapi import FastAPI
from sklearn import datasets

app = FastAPI()

iris = datasets.load_iris()
model = joblib.load("app/model.joblib")


@app.get("/")
async def root():
    return {"message": "Welcome to my iris API"}


@app.get("/data")
async def get_data():
    return {"data": iris.data.tolist(), "target": iris.target.tolist()}


@app.get("/pred")
async def get_prediction(sepal_length, sepal_width, petal_length, petal_width):
    pred = int(
        model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
    )
    pred_class = iris.target_names[pred]
    return {"class": pred, "class name": pred_class}
