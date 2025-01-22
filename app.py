# app.py
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle  
import io

app = FastAPI()

model = None
scaler = None

class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float
    Vibration: float

# function to create synthetic data
def make_fake_data():
    # create 1000 rows of data
    data = {
        'Machine_ID': [],
        'Temperature': [],
        'Run_Time': [],
        'Vibration': [],
        'Downtime_Flag': []
    }
    
    
    for i in range(1000):
        data['Machine_ID'].append(np.random.randint(1, 11))
        data['Temperature'].append(np.random.normal(75, 15))
        data['Run_Time'].append(np.random.normal(100, 30))
        data['Vibration'].append(np.random.normal(0.5, 0.2))
    
  
    df = pd.DataFrame(data)
    
    # simple rules for downtime
    df['Downtime_Flag'] = 0
    df.loc[df['Temperature'] > 90, 'Downtime_Flag'] = 1
    df.loc[df['Run_Time'] > 150, 'Downtime_Flag'] = 1
    df.loc[df['Vibration'] > 0.8, 'Downtime_Flag'] = 1
    
    return df

# upload endpoint
@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    # read the file
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
   
    df.to_csv('data.csv', index=False)
    
    return {"message": "File uploaded!", "rows": len(df)}

# training endpoint
@app.post("/train")
async def train_model():
    global model, scaler
    
    
    try:
        df = pd.read_csv('data.csv')
    except:
        print("No data found, making fake data")
        df = make_fake_data()
        df.to_csv('data.csv', index=False)
    
    # get features ready
    X = df[['Temperature', 'Run_Time', 'Vibration']]
    y = df['Downtime_Flag']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    
    accuracy = model.score(X_test, y_test)
    
    
    # save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    return {
        "message": "Model trained!",
        "accuracy": accuracy
    }

# prediction endpoint
@app.post("/predict")
async def predict(input_data: PredictionInput):
    global model, scaler
    
    
    if model is None:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    
    
    input_values = [[
        input_data.Temperature,
        input_data.Run_Time,
        input_data.Vibration
    ]]
    
    
    input_scaled = scaler.transform(input_values)
    
    
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0].max()
    
    
    return {
        "Downtime": "Yes" if prediction == 1 else "No",
        "Confidence": proba
    }

# run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)