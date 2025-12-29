# Engineering-Students-DataAnalysis
Engineering Student Performance Analysis & Prediction
Project Overview
This project investigates the determinants of academic performance decline among engineering students at Université Internationale de Casablanca (UIC). By analyzing psychosocial, behavioral, and academic data from 114 students, this study identifies key risk factors such as burnout, stress, and study habits.



The project culminates in a machine learning model capable of predicting "at-risk" students with ≈80% accuracy, creating a foundation for early-warning systems in educational environments.


Objectives:

Identify the primary drivers of academic decline (e.g., stress, sleep, "feeling lost").



Visualize these patterns using an interactive Power BI dashboard.



Predict student risk using a Random Forest classifier deployed via a Streamlit web interface.


Motivation:
This project was developed to apply concepts learned during the IBM AI Engineering Professional Certificate and coursework at UIC. It serves as a practical implementation of the Data Science lifecycle—from data cleaning and exploratory analysis to model training and deployment.

Key Findings:

Burnout is the #1 Predictor: Emotional exhaustion was a stronger indicator of decline than stress alone.



The "Cramming" Trap: Over 60% of declining students reported studying only the "day before the exam".


The Tipping Point: Risk of decline spikes significantly when a student struggles with 2 or more modules.


Gender Differences: The data highlighted distinct patterns in stress and burnout levels between genders, suggesting the need for tailored support.

Repository Contents:

ModelTestting.ipynb: Jupyter Notebook containing data cleaning, EDA, and model training (Logistic Regression, Decision Tree, Random Forest).




StudentDataDashboard.pbix: Interactive Power BI dashboard for visualizing student data.


Engineering Student Performance Data Analysis.pdf: Full research report detailing hypotheses, methodology, and results.


app.py: (You will need to create this) The Python script for the Streamlit web interface.


EngStudent_Db(Eng_Stats).csv: The dataset used for analysis

 How to Run the Streamlit App:
This project includes a Streamlit web application that allows users to input student data and get a real-time risk prediction.

1. Prerequisites
Ensure you have Python installed. You will need the following libraries:
pip install streamlit pandas numpy scikit-learn , make sure to put train_model.py, app.py,model_tree.pkl and EngStudent_Db(Eng_Stats).csv in one folder.
2. Path Configuration :
Crucial Step: Before running the app, you must update the file path in the code to point to where the dataset or model is saved on your machine.
Open app.py (or your Streamlit script) and look for the line loading the model or data:

# CHANGE THIS PATH to your local file location
df = pd.read_csv("C:/Users/YourName/Documents/Project/EngStudent_Db(Eng_Stats).csv", sep=";")
If you are just loading a saved model (.pkl file), ensure that path is correct as well.

3. Running the App
Open your terminal or command prompt, navigate to the project folder, and run:

python -m streamlit run app.py

(Replace app.py with the actual name of your Python script if it's different).

This will launch the application in your default web browser (usually at http://localhost:8501).

Hope This Helps! 
I hope this project provides value to other students or educators interested in Educational Data Mining. Feel free to fork this repository, submit issues, or reach out if you have questions!

Disclaimer: This tool is for research demonstration purposes only. Predictions should be interpreted probabilistically and used to support, not label, students.
