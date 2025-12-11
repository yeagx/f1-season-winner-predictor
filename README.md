# 🏎️ F1 Season Winner Predictor

A machine learning web application that predicts the Formula 1 season champion using XGBoost classifier.

## Features

- **Data Processing**: Aggregates F1 race data by driver and season
- **Machine Learning Model**: Uses XGBoost to predict season winners
- **Interactive UI**: Built with Streamlit for easy interaction
- **Feature Importance**: Visualizes which features matter most for predictions
- **Next Season Prediction**: Predicts the winner of the next season based on the latest data

## Requirements

- Python 3.7+
- F1.xlsx dataset file

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure `F1.xlsx` is in the same directory as `app.py`
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. The app will open in your browser automatically

## How It Works

1. Loads F1 race data from `F1.xlsx`
2. Processes data to create driver-season aggregates (points, pit stops, positions, etc.)
3. Trains an XGBoost classifier to predict champions
4. Displays model accuracy and feature importance
5. Predicts the next season's winner with probability scores

## Model Features

The model uses the following features:
- Number of races
- Average finish position
- Average points per race
- Total points
- Average pit stops
- Average pit time
- Driver aggression score
- Position gains
- Constructor (encoded)

