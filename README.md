# Manufacturing Predictive Analysis system

## How to Use

1. Install the dependencies:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn python-multipart
```

2. Run the program:
```bash
python -m uvicorn app:app --reload
```

3. The website will be at http://localhost:8000

## What it Does

### Upload Data (/upload)
You can upload a CSV file with machine data. Make sure it has these columns:
- Machine_ID
- Temperature
- Run_Time
- Vibration
- Downtime_Flag

Example:
```bash
curl -X POST -F "file=@your_data.csv" http://localhost:8000/upload
```

### Train the Model (/train)
This will train the machine learning model. If you don't have data, it'll create synthetic data.

Example:
```bash
curl -X POST http://localhost:8000/train
```

### Make Predictions (/predict)
Predict if it might break down.

Example:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"Temperature": 85.5, "Run_Time": 130.2, "Vibration": 0.75}' http://localhost:8000/predict
```

## Notes
- Made with FastAPI.
- Uses Random Forest.
- Saves everything in CSV and pickle files
- Probably needs more error checking but it works! 🎉
