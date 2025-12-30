---
layout: post
title: "Weather Dashboard – Streamlit Web Application"
image: "/posts/weather.png"
tags: [Python, Streamlit, REST API, OpenWeatherMap, WebApp, Data]
---

This project presents a **Streamlit-based Weather Dashboard** that retrieves and displays real-time weather information using the **OpenWeatherMap API**.  
The application focuses on clean API integration, secure configuration management, and rapid development of interactive web interfaces with Python.

---

# Table of Contents
- [00. Project Overview](#overview)
  - [Context](#context)
  - [Actions](#actions)
  - [Results](#results)
  - [Growth / Next Steps](#growth)
- [01. System Design Overview](#system-design)
- [02. Core Application Code](#code)
- [03. Streamlit UI & Interaction](#ui)
- [04. Error Handling](#errors)
- [05. Discussion](#discussion)

---

# 00. Project Overview <a name="overview"></a>

## Context <a name="context"></a>
Access to up-to-date weather information is a common requirement for dashboards and decision-support applications.  
The goal of this project was to build a **lightweight, deployable, and user-friendly weather application** without relying on traditional frontend frameworks.

---

## Actions <a name="actions"></a>
I developed a Streamlit web application that:
- Accepts a city name as user input  
- Sends HTTP requests to the OpenWeatherMap REST API  
- Retrieves and parses real-time weather data  
- Displays key weather attributes instantly in a clean UI  
- Securely manages the API key using environment variables  

---

## Results <a name="results"></a>
- Real-time weather data retrieval  
- Clean and responsive user interface  
- Secure API key handling  
- Minimal and readable codebase  
- Successfully deployed on Streamlit Cloud  

Live Demo:  
https://weatherdashboard-cj4y8yhnuubjt9aacbeaav.streamlit.app/

---

## Growth / Next Steps <a name="growth"></a>
Future enhancements include:
- Multi-day weather forecasts  
- Weather icons and visual indicators  
- Improved input validation  
- Location-based weather detection  

---

# 01. System Design Overview <a name="system-design"></a>

User  
→ Streamlit UI  
→ OpenWeatherMap API  
→ JSON Response  
→ Data Parsing  
→ Weather Dashboard Output  

Core components include Streamlit for the interface, Requests for HTTP communication, and OpenWeatherMap as the data provider.

---

# 02. Core Application Code <a name="code"></a>

import streamlit as st  
import requests  
import os  

API_KEY = os.getenv("OPENWEATHER_API_KEY")  
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"  

st.set_page_config(page_title="Weather Dashboard", page_icon="🌤")  

st.title("🌤 Weather Dashboard")  

city = st.text_input("Enter city name")  

if st.button("Get Weather"):  
    if city:  
        params = {  
            "q": city,  
            "appid": API_KEY,  
            "units": "metric"  
        }  

        response = requests.get(BASE_URL, params=params)  

        if response.status_code == 200:  
            data = response.json()  

            st.subheader(data["name"])  
            st.write(f"🌡 Temperature: {data['main']['temp']} °C")  
            st.write(f"💧 Humidity: {data['main']['humidity']} %")  
            st.write(f"🌬 Wind Speed: {data['wind']['speed']} m/s")  
            st.write(f"☁ Description: {data['weather'][0]['description']}")  
        else:  
            st.error("City not found")  

---

# 03. Streamlit UI & Interaction <a name="ui"></a>

The user interface is built entirely with Streamlit components.  
A text input field captures the city name, and a button explicitly triggers the API request.  
This prevents unnecessary API calls and keeps the application efficient and responsive.

---

# 04. Error Handling <a name="errors"></a>

The application validates API responses using HTTP status codes.  
If an invalid city name is provided or the request fails, a clear error message is displayed instead of crashing the application.

---

# 05. Discussion <a name="discussion"></a>

This project demonstrates:
- Real-world REST API consumption  
- Secure handling of sensitive configuration data  
- Clean separation of logic and presentation  
- Rapid development and deployment using Streamlit  

It serves as a strong **portfolio-ready example** of modern Python web development.

---

Live Demo:  
https://weatherdashboard-cj4y8yhnuubjt9aacbeaav.streamlit.app/

GitHub Repository:  
https://github.com/LShahmiri/Weather_Dashboard

