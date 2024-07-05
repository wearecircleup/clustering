# Customer Segmentation API (FastAPI with K-Means Clustering)

This project provides a RESTful API for customer segmentation using a pre-trained K-Means clustering model. The API allows you to input new customer data and receive predictions about the cluster to which each customer belongs.

## Features

- **Clustering Prediction:** Predict the cluster membership of new customer data based on their behavioral patterns and attributes.
- **FastAPI:** Built using FastAPI, a modern, high-performance Python web framework for building APIs.
- **Dockerized:** The API is containerized using Docker for easy deployment and reproducibility.
- **Robust Data Processing:**  Handles missing values and applies the same preprocessing steps used during model training.

## **Getting Started**

### **Prerequisites**

- Docker installed on your machine.

### **Running the API**

**Clone the Repository**
 ```bash
 git clone https://your-repository-url.git
 cd customer-segmentation-api
 ```

**Build the Docker Image**

```bash
docker build -t customer-segmentation-api .
```

**Run the Docker Container**

```bash
docker run -d -p 8000:8000 --name customer-segmentation-container customer-segmentation-api
```

### **API Endpoint** `/predict (POST)`

**Input** A JSON object containing customer data with the following fields

```py
{
    "fullVisitorId": "string",
    "channelGrouping": "string",
    "weekend_prop": 0,
    "hour": 0,
    "sessionId": 0,
    "device.browser": "string",
    "device.deviceCategory": "string",
    "device.isMobile": 0,
    "device.operatingSystem": "string",
    "totals.hits": 0,
    "totals.pageviews": 0,
    "bounce_prop": 0,
    "trafficSource.medium": "string"
}
```

**Output** A JSON object with the predicted cluster

```py
  {
      "cluster": 2 
  }
```
### **Testing the API**

```bash
curl -X POST http://localhost:8000/predict \
     -H 'Content-Type: application/json' \
     -d '{
          "fullVisitorId": "0213131142648941",
          "channelGrouping": "Direct",
          "weekend_prop": 0.0,
          "hour": 22.0,
          "sessionId": 1,
          "device.browser": "Chrome",
          "device.deviceCategory": "desktop",
          "device.isMobile": 0.0,
          "device.operatingSystem": "Macintosh",
          "totals.hits": 14.0,
          "totals.pageviews": 13.0,
          "bounce_prop": 0.0,
          "trafficSource.medium": "(none)"
       }'
```

### **Model Details**
The K-Means model is pre-trained on a dataset of Google Analytics customer behavior data.
The preprocessing steps involve scaling numerical features, one-hot encoding categorical features, and feature selection.
The model file (kmeans_model.joblib) and preprocessor/selector files need to be placed in the app/models/ directory before building the Docker image.
The K-Means model is pre-trained on a dataset of Google Analytics customer behavior data.
The preprocessing steps involve scaling numerical features, one-hot encoding categorical features, and feature selection.
The model file (kmeans_model.joblib) and preprocessor/selector files need to be placed in the app/models/ directory before building the Docker image.