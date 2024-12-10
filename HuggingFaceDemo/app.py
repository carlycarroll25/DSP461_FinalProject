##### LOAD LIBRARIES #####

import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

##### DATA UPLOADING AND PREPROCESSING #####

# get the data from the github repository
url = "https://raw.githubusercontent.com/carlycarroll25/DSP461_FinalProject/refs/heads/main/Data/affordability.csv"

#load in affordability data
affordability = pd.read_csv(url)
print(affordability.shape)


### FEATURE ENGINEERING AND SCALING ###

# define regression features and target
regression_features = [
    "HousingCostAvg", "TotalLivingCost", "median_family_income",
    "TotalPop", "crime_rate_per_100000"
]
X_reg = affordability[regression_features]
y_reg = affordability["AffordabilityScore"]

#Find the min and max of each feature in X_reg
min_max = {}
for feature in regression_features:
    min_max[feature] = (X_reg[feature].min(), X_reg[feature].max())
print("X_reg limits:", min_max)

# define classification features and target
classification_features = [
    "INflow", "OUTflow", "TotalPop", "HousingCostAvg", "median_family_income"
]
X_class = affordability[classification_features]
if "MigrationClass" not in affordability.columns:
    affordability["MigrationClass"] = pd.cut(
        affordability["NET in"], bins=[-float('inf'), -1, 1, float('inf')],
        labels=["Net Loss", "Neutral", "Net Gain"]
    )
y_class = affordability["MigrationClass"]

#Find the min and max of each feature in X_class
min_max = {}
for feature in classification_features:
    min_max[feature] = (X_class[feature].min(), X_class[feature].max())
print("X_class limits:", min_max)

## scale features
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)
X_class_scaled = scaler.fit_transform(X_class)

## split the data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_scaled, y_reg, test_size=0.3, random_state=42
)
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class_scaled, y_class, test_size=0.3, random_state=42
)

# separate scalers for affordability and migration
scaler_affordability = StandardScaler()
scaler_migration = StandardScaler()

# fit scalers on respective datasets
X_affordability = affordability[["HousingCostAvg", "TotalLivingCost", "median_family_income", "TotalPop", "crime_rate_per_100000"]]
X_migration = affordability[["INflow", "OUTflow", "TotalPop", "HousingCostAvg", "median_family_income"]]

scaler_affordability.fit(X_affordability)
scaler_migration.fit(X_migration)

### MODEL TRAINING ###

# train regression model
regressor = RandomForestRegressor(random_state=42, n_estimators=100)
regressor.fit(X_train_reg, y_train_reg)

# train classification model
classifier = RandomForestClassifier(random_state=42, n_estimators=100)
classifier.fit(X_train_class, y_train_class)


### PREDICTION FUNCTIONS ###

# affordability prediction function
def predict_affordability(HousingCostAvg, TotalLivingCost, median_family_income, TotalPop, crime_rate_per_100000):
    features = pd.DataFrame(
        [[HousingCostAvg, TotalLivingCost, median_family_income, TotalPop, crime_rate_per_100000]],
        columns=["HousingCostAvg", "TotalLivingCost", "median_family_income", "TotalPop", "crime_rate_per_100000"]
    )
    # scale the features
    scaled_features = scaler_affordability.transform(features)
    prediction = regressor.predict(scaled_features)
    return f"Predicted Affordability Score: {prediction[0]:.2f}"

# migration classification function
def classify_migration(INflow, OUTflow, TotalPop, HousingCostAvg, median_family_income):
    features = pd.DataFrame(
        [[INflow, OUTflow, TotalPop, HousingCostAvg, median_family_income]],
        columns=["INflow", "OUTflow", "TotalPop", "HousingCostAvg", "median_family_income"]
    )
    # scale the features
    scaled_features = scaler_migration.transform(features)
    prediction = classifier.predict(scaled_features)
    return f"Predicted Migration Class: {prediction[0]}"


### GRADIO SCRIPT ###

# create the gradio interface
with gr.Blocks() as demo:
    # regression model
    title = gr.HTML("<h1><center>Regression Affordability Predictor</center></h1>")
    gr.Interface(fn=predict_affordability, 
                inputs=[gr.Slider(30000,125000), gr.Slider(50000,175000),gr.Slider(45000,175000),gr.Slider(9000,10000000),gr.Slider(20,1200)], 
                outputs="text")
    # classification model
    title = gr.HTML("<h1><center>Migration Classification Predictor</center></h1>")
    gr.Interface(fn=classify_migration, 
                inputs=[gr.Slider(600,300000), gr.Slider(500,325000),gr.Slider(9000,10000000),gr.Slider(30000,125000), gr.Slider(50000,175000)], 
                outputs="text")
demo.launch()