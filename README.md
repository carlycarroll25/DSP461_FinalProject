# Predicting Housing Affordability and Migration Patterns

## Overview
This project utilizes machine learning techniques and interactive tools to analyze and predict housing affordability and migration patterns across US counties. Using socioeconomic, demographic, and migration data, we developed models for clustering, regression, and classification. Additionally, interactive tools were implemented to make the findings accessible to users, enabling them to explore scenarios and gain actionable insights.

## Project Structure

### Files
- **`ExploratoryDataAnalysis.ipynb`**: Contains the exploratory data analysis code, including visualizations and basic statistical insights.
- **`Experiment1_DatClustering.ipynb`**: Implements clustering techniques, including K-means and hierarchical clustering.
- **`Experiment2_PredictiveModeling.ipynb`**: Implements predictive modeling techniques such as regression and classification.
- **`Experiment3_NeuralNetworkModels.ipynb`**: Applies neural networks for regression and classification tasks with cross validation and feature importance.
- **`InteractiveTools.ipynb`**: Provides interactive tools for exploring model predictions and insights, including Gradio interfaces.

### Reports
- **`DSP461_FinalProjectReport.pdf`**: A comprehensive report detailing the methodology, analysis, results, and conclusions.
- **`DSP461_FinalProjectPresentation.pdf`**: A presentation summarizing the key findings and methodology.

## Methodology
The project utilized the following steps:
1. **Data Preprocessing**: Cleaned, merged, and standardized datasets to calculate affordability scores and migration statistics.
2. **Exploratory Data Analysis**: Investigated data distributions and relationships using statistical graphs and metrics.
3. **Clustering**: Applied K-means and hierarchical clustering to uncover patterns in affordability and migration.
4. **Predictive Modeling**: Used regression and classification techniques to predict affordability scores and migration categories.
5. **Interactive Tools**: Developed user-friendly tools using Gradio to explore model results interactively.

## Key Results
- **Regression**: The affordability prediction model achieved a mean absolute error (MAE) of 0.14.
- **Classification**: Migration prediction models achieved up to 77% accuracy with tuned Random Forest classifiers.
- **Clustering**: Identified three distinct clusters based on affordability characteristics, validated using silhouette scores.

## Interactive Tools
Interactive tools were developed using Gradio:
1. **Affordability Prediction Interface**: Predicts affordability scores based on input features.
2. **Migration Classification Interface**: Categorizes migration trends into Net Gain, Net Loss, or Neutral.
3. **Interactive Demo
Experience the project's features interactively with our Hugging Face demo. Click the link below to explore:
[Hugging Face Demo](https://huggingface.co/spaces/22tsangr/demo)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/YourRepositoryName.git
