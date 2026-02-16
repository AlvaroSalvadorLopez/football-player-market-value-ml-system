# ⚽ Football Player Market Value Prediction – Machine Learning System

This project was developed as part of my Master's degree in Computer Science and presents a complete end-to-end machine learning system designed to predict the market value of professional football players using real-world performance data.

The system implements, trains, evaluates, and compares multiple machine learning models and provides an interactive interface for real-time predictions.

---

# 📌 Project Overview

The objective of this project is to build a robust predictive system capable of estimating football players’ market value based on their performance statistics.

This project covers the full machine learning pipeline:

• Data collection and preprocessing  
• Feature engineering and dataset preparation  
• Model training and hyperparameter tuning  
• Model evaluation and comparison  
• Deployment of an interactive prediction interface  

The system uses real-world data obtained from football statistics platforms such as Transfermarkt and FBRef.

---

# 🧠 Machine Learning Models Implemented

The following regression models were implemented and evaluated:

• Random Forest Regressor  
• XGBoost Regressor  
• CatBoost Regressor  
• Support Vector Regression (SVR)  
• K-Nearest Neighbors (KNN)  
• Gradient Boosting Regressor  

Models were evaluated using Root Mean Squared Error (RMSE), and their performance was compared to identify the most accurate approach.

---

# ⚙️ System Architecture

The system is structured into two independent machine learning pipelines:

Project 1
• Model training and evaluation  
• Model selection based on performance  
• Prediction generation  

Project 2
• Alternative model implementations  
• Independent training pipeline  
• Performance comparison with Project 1  

Each project includes:

• Training script  
• Prediction script  
• Model storage  
• Streamlit interface  

---

# 📊 Key Features

• Complete end-to-end machine learning pipeline  
• Comparative analysis of multiple regression models  
• Real-world dataset processing  
• Model performance evaluation using RMSE  
• Interactive prediction interface built with Streamlit  
• Modular and scalable system design  

---

# 🖥 Interactive Prediction Interface

The system includes a Streamlit application that allows users to:

• Select player performance attributes  
• Run predictions using trained models  
• View predicted market value instantly  

To run the interface:

streamlit run app_project1.py

or

streamlit run app_project2.py

---

# 🛠 Technologies Used

Programming Language:
• Python

Libraries:
• Pandas  
• NumPy  
• Scikit-learn  
• XGBoost  
• CatBoost  
• Matplotlib  
• Seaborn  
• Streamlit  

Tools:
• VSCode  
• Git  
• GitHub  

---

# 📁 Repository Structure

data/              → Dataset files  
models/            → Trained models  
project1/          → First ML system  
project2/          → Second ML system  
docs/              → Thesis documentation  
app_project1.py    → Streamlit interface (Project 1)  
app_project2.py    → Streamlit interface (Project 2)  
requirements.txt   → Dependencies  

---

# 📄 Documentation

The full thesis report is available in:

docs/thesis-report.pdf

This document includes:

• Methodology  
• Data analysis  
• Model comparison  
• Experimental results  
• System evaluation  

---

# 👨‍💻 Author

Álvaro Salvador López  

Computer Engineer  
Master’s Degree in Computer Science  

Fields of interest:

• Data Analytics  
• Machine Learning  
• Artificial Intelligence  

GitHub:  
https://github.com/AlvaroSalvadorLopez

