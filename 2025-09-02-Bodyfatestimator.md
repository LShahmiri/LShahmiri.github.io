---
layout: post
title: "Body Fat Estimator – Machine Learning Web App Deployed on Azure"
image: "/posts/bodyfat.png"
tags: [Machine Learning, Flask, Scikit-learn, Azure, Python, CI/CD]
---

This project demonstrates a complete **Machine Learning deployment pipeline** where a regression model predicts **body fat percentage** from anthropometric measurements.

The system integrates **Scikit-learn**, **Flask**, **GitHub**, and **Microsoft Azure Web App Service** to create a fully functional **cloud-hosted ML web application**.

Users can input body measurements through a web interface and instantly receive a predicted **body fat percentage**.

---

# Table of Contents

- Project Overview
- Dataset
- Machine Learning Model
- Flask Web Application
- Azure Web App Service Deployment
- CI/CD Pipeline
- Discussion
- Live Demo

---

# Project Overview

Estimating body fat percentage normally requires specialized laboratory equipment such as **DEXA scans or hydrostatic weighing**.

Machine learning allows body fat estimation using simple **anthropometric measurements**, making prediction accessible through a web application.

This project demonstrates how a machine learning model can be transformed into a **production-ready cloud service**.

The pipeline includes:

• Training a regression model  
• Saving the trained model using pickle  
• Building a Flask web application  
• Creating HTML templates for user input  
• Deploying the application to Azure Web App Service  
• Connecting the repository to GitHub for automated deployment  

The final system allows users to enter body measurements and receive **instant predictions of body fat percentage**.

---

# Dataset

The model was trained using the **Body Fat Prediction Dataset**.

Dataset source:

https://www.kaggle.com/datasets/fedesoriano/body-fat-prediction-dataset

The dataset contains anthropometric measurements collected from adult male subjects.

Example variables used:

Density – Body density measurement  
Abdomen – Abdomen circumference  
Chest – Chest circumference  
Weight – Body weight  
Hip – Hip circumference  

These features are used to estimate body fat percentage.

---

# Machine Learning Model

A regression model was trained using **Scikit-learn**.

The trained model is saved using **pickle** so it can be loaded inside the web application.

Example model loading:

```python
import pickle

file1 = open('bodyfatmodel.pkl','rb')
rf = pickle.load(file1)
file1.close()
```

The model receives body measurements and predicts body fat percentage.

---

# Flask Web Application

The machine learning model is integrated into a **Flask web application**.

Users input measurements in a web form and the prediction is returned instantly.

```python
from flask import Flask, request, render_template
import pickle
import pandas as pd

file1 = open('bodyfatmodel.pkl', 'rb')
rf = pickle.load(file1)
file1.close()

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def predict():

    if request.method == 'POST':

        density = float(request.form['density'])
        abdomen = float(request.form['abdomen'])
        chest = float(request.form['chest'])
        weight = float(request.form['weight'])
        hip = float(request.form['hip'])

        input_features = pd.DataFrame(
            [[density, abdomen, chest, weight, hip]],
            columns=["Density","Abdomen","Chest","Weight","Hip"]
        )

        prediction = rf.predict(input_features)[0].round(2)

        string = "Estimated Body Fat Percentage: " + str(prediction) + "%"

        return render_template('show.html', string=string)

    return render_template('home.html')

if __name__ == "__main__":
    app.run(debug=True)
```

---

# Azure Web App Service Deployment

The application is deployed using **Microsoft Azure Web App Service**.

Azure Web App Service provides:

• Managed hosting for web applications  
• Built-in Python runtime  
• Secure HTTPS hosting  
• Easy integration with GitHub  
• Automatic deployment  

Deployment architecture:

User Browser  
↓  
Azure Web App Service  
↓  
Flask Application (Gunicorn)  
↓  
Machine Learning Model  

The application runs inside a **Linux-based Azure environment** using a Python runtime.

---

# Procfile

Azure uses **Gunicorn** as the production server.

```
web: gunicorn app:app
```

---

# CI/CD Pipeline

The repository is connected to **GitHub**.

Whenever code is pushed:

1. GitHub builds the environment  
2. Dependencies are installed  
3. The application is automatically deployed to Azure  

Example installation command:

```bash
pip install -r requirements.txt
```

Example requirements:

```bash
flask
pandas
scikit-learn
gunicorn
```

---

# Discussion

This project demonstrates the **complete lifecycle of deploying a machine learning model**:

• Model training  
• Model serialization  
• Web application development  
• Cloud deployment  
• Continuous integration and deployment  

Future improvements could include:

• Using advanced regression models  
• Adding more features to improve accuracy  
• Adding data visualization  
• Improving the user interface  

---

# Live Demo

👉 [Demo](https://bodyfat-anfse0brekfjg7ap.canadacentral-01.azurewebsites.net/)

---

# GitHub Repository

👉 [GitHub](https://github.com/LShahmiri/Body-Fat-Estimator)
