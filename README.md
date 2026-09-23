# 🧠 Student Mental Health Prediction

A Machine Learning based web application that predicts a student's **Mental Health Score** using academic, social media usage, lifestyle, and stress-related factors.

The project provides an interactive web interface where users can enter their information and get a predicted mental health score through a FastAPI backend connected to a trained Machine Learning model.

---

## 📌 Project Overview

Social media usage, study patterns, sleep, physical activity, and stress levels can be related to students' mental well-being.

This project uses Machine Learning to predict a student's **Mental Health Score** based on different personal, academic, digital-usage, and lifestyle-related features.

The application consists of:

- 🧠 Machine Learning model
- ⚡ FastAPI backend
- 🌐 HTML/CSS/JavaScript frontend
- 📊 Interactive prediction form
- 📈 Mental Health Score result display

---

## ✨ Features

- Student information input form
- Social media usage analysis
- Study and lifestyle information
- Stress level selection
- Machine Learning based prediction
- FastAPI REST API
- Interactive web interface
- Predicted Mental Health Score display
- Responsive and clean frontend design

---

## 🛠️ Technologies Used

### Machine Learning
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

### Backend
- FastAPI
- Uvicorn
- Pydantic

### Frontend
- HTML5
- CSS3
- JavaScript

### Development Tools
- Jupyter Notebook
- VS Code
- Git
- GitHub
- uv / Python virtual environment

---

## 📊 Input Features

The model uses the following features:

| Feature | Description |
|---|---|
| Age | Student's age |
| Gender | Student's gender |
| Country | Student's country |
| Academic Level | Undergraduate, Graduate, or High School |
| Most Used Platform | Social media platform used most |
| Purpose of Use | Main purpose of social media usage |
| Average Daily Usage Hours | Daily social media usage |
| Daily Unlocks | Number of daily device/social media unlocks |
| Study Hours | Daily study hours |
| Physical Activity Hours | Daily physical activity |
| Sleep Hours Per Night | Average sleep duration |
| Stress Level | Low, Medium, High, or Very High |

---

## 🤖 Machine Learning

The target variable of the project is:

**`Mental_Health_Score`**

The data preprocessing includes numerical, ordinal, and categorical feature processing.

`Study_Hours` is transformed using logarithmic transformation and scaling.

`Stress_Level` is treated as an ordinal feature, while categorical variables such as gender, academic level, platform, purpose of use, and grouped country are processed as categorical features.

The trained model is saved as:

```text
Student_Social_Media_And_Mental_Health_Impact.pkl
