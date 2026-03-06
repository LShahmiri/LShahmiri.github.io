---
layout: post
title: "Financial Transactions Analytics Dashboard – Power BI"
image: "/posts/PowerBI.png"
tags: [Power BI, Data Analytics, Dashboard, Data Visualization, Business Intelligence]
---

This project presents an interactive **Financial Transactions Analytics Dashboard** built with **Power BI**, designed to explore transaction behaviour, financial flows, and balance patterns across multiple banks, cities, and payment methods.

The dashboard enables **dynamic filtering, trend analysis, and multi-dimensional financial exploration**, supporting data-driven decision making.

---

# Table of Contents
- [00. Project Overview](#overview)
  - [Context](#context)
  - [Actions](#actions)
  - [Results](#results)
- [01. Dashboard Design](#design)
- [02. Dataset Structure](#dataset)
- [03. Key Analytics Features](#features)
- [04. Business Insights](#insights)
- [05. Technologies Used](#tech)
- [06. Discussion](#discussion)
- [Dashboard Preview](#preview)

---

# 00. Project Overview <a name="overview"></a>

## Context <a name="context"></a>

Financial institutions process large volumes of transactions daily.  
However, understanding transaction behaviour across customers, cities, and payment systems requires effective **data visualisation and interactive analytics**.

Organizations need tools that allow them to:

- Analyse transaction behaviour over time
- Monitor account balances and financial flows
- Compare activity across regions and payment channels
- Identify unusual patterns in transaction activity

---

## Actions <a name="actions"></a>

To address these needs, I designed an **interactive Power BI dashboard** that:

- Integrates multiple transaction attributes
- Supports dynamic filtering
- Visualises transaction trends across time
- Provides multi-dimensional financial insights
- Enables quick exploration of behavioural patterns

The dashboard allows analysts and decision makers to quickly explore large transaction datasets.

---

## Results <a name="results"></a>

The dashboard enables:

- Interactive exploration of transaction data
- Identification of seasonal transaction trends
- Comparison of transaction behaviour across cities
- Monitoring of account balances
- Flexible filtering across multiple transaction attributes

---

# 01. Dashboard Design <a name="design"></a>

The dashboard was designed using **Power BI Desktop**, focusing on usability and analytical clarity.

The interface includes:

Interactive filters for:

- Bank Name (Sender / Receiver)
- City
- Device Type
- Customer Gender
- Age Group
- Merchant Name
- Payment Method
- Transaction Type
- Transaction Purpose

These filters allow users to explore transaction behaviour across multiple dimensions.

<p align="center"> <img width="1212" height="235" alt="filter" src="https://github.com/user-attachments/assets/57f45636-cba0-4f22-b1db-9b81f647326b" />
</p>


---

# 02. Dataset Structure <a name="dataset"></a>

The dataset contains financial transaction records with attributes including:

- BankNameSent
- BankNameReceived
- City
- DeviceType
- MerchantName
- PaymentMethod
- TransactionType
- Purpose
- Customer demographics (Age Group, Gender)
- Transaction Amount
- Remaining Balance
- Currency

The dataset supports multi-dimensional financial analysis across geographic and behavioural categories.

---

# 03. Key Analytics Features <a name="features"></a>

### Transaction Trends

A **monthly transaction trend analysis** visualises fluctuations in transaction activity throughout the year.

Users can switch between multiple visualisation modes:

- Line chart (Transaction Amount)
- Column chart (Transaction Amount)
- Line chart (Account Balance)
- Column chart (Account Balance)
<p align="center"> <img width="1183" height="482" alt="line chart" src="https://github.com/user-attachments/assets/2a6e8e42-a615-4f9c-9834-8fb47a77ab6e" /> </p>

<p align="center"> <img width="1262" height="485" alt="column chart" src="https://github.com/user-attachments/assets/33f1a0f9-e444-494a-b343-bdbebeb598b3" />
</p>

---

### Multi-City Comparison

The dashboard compares financial activity across multiple cities:

- Bangalore
- Delhi
- Hyderabad
- Mumbai

Each city view includes:

- Transaction Amount
- Remaining Account Balance
- Currency-specific analysis

This enables geographic comparison of financial activity.

<img width="1035" height="580" alt="Multicity" src="https://github.com/user-attachments/assets/73f57638-a4a9-469f-9f4c-c9b665ab79d2" />

---

### Financial Balance Monitoring

The system tracks **remaining balances alongside transaction amounts**, allowing analysts to identify financial flow patterns and liquidity changes over time.

---

# 04. Business Insights <a name="insights"></a>

The dashboard helps uncover insights such as:

- Seasonal variations in transaction activity
- Differences in transaction behaviour across cities
- Payment method preferences
- Merchant activity patterns
- Transaction patterns across customer demographics

These insights can support **banking analytics, financial monitoring, and operational decision making**.

---

# 05. Technologies Used <a name="tech"></a>

- Power BI Desktop
- Data Modelling
- Interactive Dashboards
- Data Visualization
- Business Intelligence Analytics

---

# 06. Discussion <a name="discussion"></a>

This project demonstrates how interactive dashboards can transform raw financial transaction data into meaningful insights.

By combining **data modelling, filtering mechanisms, and visual analytics**, the system provides an intuitive way to explore complex financial datasets.

Potential future improvements include:

- Real-time data streaming
- Fraud detection analytics
- Integration with machine learning models
- Predictive transaction analysis

---

### Dashboard Preview <a name="preview"></a>

👉 [Power BI Dashboard](https://github.com/LShahmiri/Financial-Transactions-Analytics-Dashboard-Power-BI-/blob/main/Financial%20Transactions%20Analytics%20Dashboard.pdf)

---

### 🔗 GitHub Repository

👉 [Financial-Transactions-Dashboard](https://github.com/LShahmiri/Financial-Transactions-Analytics-Dashboard-Power-BI-)
