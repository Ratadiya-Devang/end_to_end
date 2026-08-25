import pandas as pd 
import joblib 
from fastapi import FastAPI 



api = FastAPI()
model = joblib.load("model.pkl")


@api.get("/")
def testapi():
    return {"msg":"API run successfully"}


@api.post("/placement_or_not")
def predict_placement(marks:int):

    new_data = pd.DataFrame({
        "marks":[marks]
    })
    
    result = model.predict(new_data)

    result = "placement ok" if result == 1 else "no placement"

    return {"msg": result}