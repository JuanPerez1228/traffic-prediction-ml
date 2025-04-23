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

## 1.  Getting the Project onto Your Computer (Clone)

### Option A – Using VS Code (easiest)

1. Open **VS Code**  
2. Press **Ctrl + Shift + P** (or ⌘ + Shift + P on Mac) → type **Git: Clone**  
3. Paste [the repository URL your teammate gave you, e.g.](https://github.com/your-username/traffic-prediction-ml.git) 
4. Choose **any folder** (e.g., “Documents\Projects”).  
5. Click **Open** when VS Code asks if you want to open the cloned repo.
# Create a virtual envirionment. In the bottom right of your VS code terminal there are powershell, CMD, and python. 

# For CMD run:
python -m venv venv
.\venv\Scripts\activate.bat
# If CMD does not work then try the next step

# For powershell run:
python -m venv venv
.\venv\Scripts\Activate.ps1
# if this still does not work then try and ask chat gpt to walk you through it

# With your virtual environment install all packages:
pip install -r requirements.txt

# To load and inspect the dataset run:
python src/load_and_explore.py

# To train the model with hyperparameter tuning, evaluate its performance, visualize results, and save the tuned model, run:
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
