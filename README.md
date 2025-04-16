# Traffic Prediction ML Project

## Overview

This project builds a machine learning model to predict traffic congestion levels (Low, Medium, High) using various real-time and contextual factors. The model leverages features such as:
- **Vehicle count**
- **Traffic speed**
- **Road occupancy**
- **Weather conditions**
- **Traffic light status**
- **Sentiment score**
- and more...

A Random Forest classifier is used, and hyperparameter tuning is performed via GridSearchCV to achieve optimal performance. The final tuned model is saved and can be loaded for future predictions.

## Project Structure


## Setup Instructions

Follow these steps to set up and run the project:

### 1. Clone the Repository

Open your terminal and run:
```bash
git clone https://github.com/your-username/traffic-prediction-ml.git
cd traffic-prediction-ml
#For CMD run:
python -m venv venv
.\venv\Scripts\activate.bat

# For powershell run:
python -m venv venv
.\venv\Scripts\Activate.ps1

#With your virtual environment install all packages:
pip install -r requirements.txt

#To load and inspect the dataset run:
python src/load_and_explore.py

#To train the model with hyperparameter tuning, evaluate its performance, visualize results, and save the tuned model, run:
python src/train_model.py
# The script performs the following steps:

# Preprocessing: Encodes categorical variables and splits data into training and testing sets.
# Hyperparameter Tuning: Uses GridSearchCV to determine the best parameters for the Random Forest model.
# Evaluation: Prints a classification report, plots feature importances and a confusion matrix.
# Model Saving/Loading: Saves the best tuned model as traffic_rf_best_model.pkl and verifies its performance by loading it back.

# After running the training script, you will obtain:
# Confusion Matrix: Visual representation of prediction performance.
# Feature Importances: A ranked list and bar chart showing which features are most influential.
# Classification Report: Precision, recall, and F1-scores for each traffic condition class.
