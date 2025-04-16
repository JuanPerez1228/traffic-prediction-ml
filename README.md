# traffic-prediction-ml
# Traffic Prediction ML Project

This project predicts traffic conditions (Low, Medium, High) using various real-time and contextual factors like:
- Vehicle count
- Speed
- Weather
- Traffic light status
- Sentiment score and more...

# Structure

# Traffic Prediction Model

## Overview
This project builds a Random Forest model to predict traffic congestion levels (Low, Medium, High) using features such as vehicle count, traffic speed, and road occupancy. The model is tuned using GridSearchCV to find the best hyperparameters.

## Setup
1. Clone the repository:

2. Navigate to the project folder:

3. Create and activate the virtual environment:
- **Command Prompt:**
  ```
  .\venv\Scripts\activate.bat
  ```
- **PowerShell:**
  ```
  .\venv\Scripts\Activate.ps1
  ```
4. Install dependencies:


## Running the Model
- To train the model and perform hyperparameter tuning:

- The tuned model is saved as `traffic_rf_best_model.pkl`.

## Evaluation
- **Confusion Matrix:** Visualizes model prediction performance.
- **Feature Importances:** Highlights which features are most influential.
- **Classification Report:** Provides precision, recall, and F1-scores.

## Loading the Saved Model
To load the saved model in another script:
```python
import joblib
loaded_model = joblib.load("traffic_rf_best_model.pkl")
# Use loaded_model.predict() to make predictions


### 5.3. **Commit and Push Your Changes**

Finally, commit your latest code and documentation to GitHub:

1. **Stage all changes:**
   ```bash
   git add .
