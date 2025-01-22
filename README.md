# Manufacturing Predictor API

A FastAPI-based REST API for predicting machine downtime in manufacturing operations using machine learning.

## Features

- Data upload endpoint for CSV files
- Model training endpoint with performance metrics
- Prediction endpoint for real-time machine downtime prediction
- Synthetic data generation for testing
- Automatic data validation and error handling

## Requirements

```
fastapi==0.104.0
uvicorn==0.23.2
pandas==2.1.1
numpy==1.25.2
scikit-learn==1.3.0
python-multipart==0.0.6
joblib==1.3.2
pydantic==2.4.2
```

## Usage

1. Start the server:
```bash
uvicorn app:app --reload
```

2. The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Upload Data (POST /upload)
Upload manufacturing data in CSV format.

```bash
curl -X POST -F "file=@data.csv" http://localhost:8000/upload
```

Expected CSV format:
```
Machine_ID,Temperature,Run_Time,Vibration,Downtime_Flag
1,75.2,120.5,0.45,0
2,92.1,155.3,0.82,1
```

### 2. Train Model (POST /train)
Train the machine learning model on uploaded data.

```bash
curl -X POST http://localhost:8000/train
```

Response:
```json
{
    "message": "Model trained successfully",
    "metrics": {
        "accuracy": 0.945,
        "f1_score": 0.923
    }
}
```

### 3. Predict (POST /predict)
Make predictions for new data.

```bash
curl -X POST -H "Content-Type: application/json" \
    -d '{"Temperature": 85.5, "Run_Time": 130.2, "Vibration": 0.75}' \
    http://localhost:8000/predict
```

Response:
```json
{
    "Downtime": "Yes",
    "Confidence": 0.892
}
```

## Testing

The API includes synthetic data generation for testing purposes. If no data is uploaded, the system will automatically generate sample data when training the model.

## Error Handling

The API includes comprehensive error handling for:
- Invalid file formats
- Missing required columns
- Invalid data types
- Server errors
