# [VateX – eXtend the Edge](https://github.com/lastime1650/VateX)

<div align="center">
  <img
    src="https://github.com/lastime1650/VateX/blob/main/images/VATEX.png"
    alt="VATEX LOGO"
    width="500"
  />
</div>

---

# VateX Series - VateX NOVA AI

<div align="center">
  <img
    src="https://github.com/lastime1650/VateX/blob/mainv2/images/VATEX_NOVA_AI.png"
    alt="VATEX NOVA"
    width="400"
  />
</div>

---

## Structure

![initial](https://github.com/lastime1650/VATEX_NOVA_AI/blob/main/VATEX_NOVA_AI_v2.png)

---

**NOVA** means "the NEW"
It provides the capability to train and predict frameworks on deep learning and AI very easily with only RestAPI(Json).

---

## Key Components

1. **Only RestAPI**
2. **Easy Connect to AI/DNN ! @Created Simple Framework for Easy**

---

## Examples

> [!NOTE]
> 
> Python Example -> [https://github.com/lastime1650/VATEX_NOVA_AI/blob/main/api_request_client_sample.py](https://github.com/lastime1650/VATEX_NOVA_AI/blob/main/api_request_client_sample.py)
>

---

# VATEX NOVA AI - API Usage Guide

This guide provides instructions and examples for using the VATEX NOVA AI API for Deep Learning and Machine Learning tasks.

## API Endpoint Configuration

Before making requests, configure your client with the server's IP and port.

-   **{{ Sample Connection }}**
-   **IP Address:** `192.168.1.205`
-   **Port:** `10302`
-   **Base URL:** `http://192.168.1.205:10302`

### API Paths

| Task | Method | Path |
| :--- | :--- | :--- |
| **Deep Learning Train** | `POST` | `/api/solution/util/nova/DL/train` |
| **Deep Learning Predict** | `POST` | `/api/solution/util/nova/DL/predict` |
| **Machine Learning Train** | `POST` | `/api/solution/util/nova/ML/train` |
| **Machine Learning Predict** | `POST` | `/api/solution/util/nova/ML/predict` |

### [+] **WithId** API Paths
| Task | Method | Path |
| :--- | :--- | :--- |
| **WithId Samples Appending** | `POST` | `/api/solution/util/nova/with_id/sample/push` |
| **WithId Sample y Edit** | `POST` | `/api/solution/util/nova/with_id/sample/y/edit` |
| **WithId Sample X Edit** | `POST` | `/api/solution/util/nova/with_id/sample/x/edit` |
| **WithId Sample Remover** | `POST` | `/api/solution/util/nova/with_id/sample/remove` |
| **WithId - Machine Learning Predict** | `POST` | `/api/solution/util/nova/with_id/ML/train` |
| **WithId - Machine Learning Predict** | `POST` | `/api/solution/util/nova/with_id/ML/predict` |
| **WithId - Deep Learning Predict** | `POST` | `/api/solution/util/nova/with_id/DL/train` |
| **WithId - Deep Learning Predict** | `POST` | `/api/solution/util/nova/with_id/DL/predict` |

> [!NOTE]
> 
> **WithId** API - Python Example -> [https://github.com/lastime1650/VATEX_NOVA_AI/blob/main/WithId_sample.py](https://github.com/lastime1650/VATEX_NOVA_AI/blob/main/WithId_sample.py)
>

---

# Deep Learning API

## 1. Classification

### 1.1. Train a Classification Model (SoftMax)

This example demonstrates how to train a neural network for a multi-class classification task.

**Endpoint:** `POST /api/solution/util/nova/DL/train`

**Request Body:**
```json
{
  "id": "classification-model-01",
  "data": {
      "X": {
          "source": [
              [10, 20, 30], [11, 22, 33], [90, 80, 70], [95, 88, 77],
              [50, 50, 50], [45, 55, 60], [12, 25, 35], [92, 81, 74]
          ]
      },
      "y": {
          "source": [ "A", "B", "C", "B", "C", "A", "C", "B" ],
          "y_type": "one_hot_encode"
      }
  },
  "train": {
      "layer": {
          "input": { "shape": [3] },
          "hiddens": [
              { "type": "dense", "units": 16, "activation_func": "relu" }
          ],
          "output": { "units": 3, "activation_func": "softmax" }
      },
      "compile": {
          "optimizer_func": "adam",
          "loss_func": "categorical_crossentropy",
          "metrics": ["accuracy"]
      },
      "fit": {
          "epochs": 10,
          "batch_size": 1,
          "validation_split": 0.25
      }
  }
}
```

**Response:**
The response is a list of JSON objects, one for each epoch, containing the training and validation metrics.

```json
[
    {
        "accuracy": 0.3333333432674408,
        "loss": 1.0891636610031128,
        "val_accuracy": 0.5,
        "val_loss": 0.9032398462295532
    },
    {
        "accuracy": 0.3333333432674408,
        "loss": 1.082886815071106,
        "val_accuracy": 0.5,
        "val_loss": 0.9078031778335571
    },
    ...
]
```

### 1.2. Predict with the Classification Model

Use the `id` of the trained model to make predictions on new data.

**Endpoint:** `POST /api/solution/util/nova/DL/predict`

**Request Body:**
```json
{
    "id": "classification-model-01",
    "data": {
        "X": {
            "source": [
                [11, 22, 33]
            ]
        }
    }
}
```

**Response:**
The output provides the class with the highest probability (`argmax`), the lowest (`argmin`), and the probabilities for all classes (`all`).

```json
{
    "status": true,
    "output": {
        "argmax": { "B": 0.33860063552856445 },
        "argmin": { "A": 0.32945770025253296 },
        "all": {
            "A": 0.32945770025253296,
            "B": 0.33860063552856445,
            "C": 0.3319416642189026
        }
    }
}
```

## 2. Regression

### 2.1. Train a Regression Model

This example shows how to train a neural network for a regression task to predict a continuous value.

**Endpoint:** `POST /api/solution/util/nova/DL/train`

**Request Body:**
```json
{
  "id": "regression-model-01",
  "data": {
      "X": {
          "source": [
              [10, 20, 30], [11, 22, 33], [90, 80, 70], [95, 88, 77],
              [50, 50, 50], [45, 55, 60], [12, 25, 35], [92, 81, 74]
          ]
      },
      "y": {
          "source": [ 1.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 2.0 ],
          "y_type": "min_max"
      }
  },
  "train": {
      "layer": {
          "input": { "shape": [3] },
          "hiddens": [
              { "type": "dense", "units": 16, "activation_func": "relu" }
          ],
          "output": { "units": 1, "activation_func": "sigmoid" }
      },
      "compile": {
          "optimizer_func": "adam",
          "loss_func": "mse",
          "metrics": ["mae"]
      },
      "fit": {
          "epochs": 10,
          "batch_size": 1,
          "validation_split": 0.25
      }
  }
}
```

**Response:**
The API returns a list of metrics for each training epoch.

```json
[
    {
        "loss": 0.2034807652235031,
        "mae": 0.38669994473457336,
        "val_loss": 0.13519415259361267,
        "val_mae": 0.3201397657394409
    },
    {
        "loss": 0.20247896015644073,
        "mae": 0.38515543937683105,
        "val_loss": 0.13282975554466248,
        "val_mae": 0.3143080770969391
    },
    ...
]
```

### 2.2. Predict with the Regression Model

**Endpoint:** `POST /api/solution/util/nova/DL/predict`

**Request Body:**
```json
{
    "id": "regression-model-01",
    "data": {
        "X": {
            "source": [
                [11, 22, 33]
            ]
        }
    }
}
```

**Response:**
The output is the single predicted numerical value.

```json
{
    "status": true,
    "output": {
        "output": 2.010890245437622
    }
}
```

---

# Machine Learning API

## 1. Classification (Random Forest)

### 1.1. Train a Classification Model

**Endpoint:** `POST /api/solution/util/nova/ML/train`

**Request Body:**
```json
{
    "id": "MyML-01",
    "data": {
        "X": { "source": [
                [10, 15, 20], [12, 18, 25], [9, 14, 19], [11, 17, 23],
                [50, 55, 60], [53, 58, 63], [47, 52, 57], [55, 60, 65],
                [90, 85, 80], [88, 82, 78], [92, 87, 83], [95, 89, 84]
            ]
        },
        "y": {
            "source": [ "A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C" ],
            "y_type": "label"
        }
    },
    "train": {
        "model": {
            "model_name": "RandomForestClassifier",
            "model_params": { "n_estimators": 100, "random_state": 4 }
        },
        "trainset": { "test_size": 0.25, "shuffle": true, "stratify": "y" }
    }
}
```

**Response:**
The `output` field represents the model's accuracy score on the test set.

```json
{
    "status": true,
    "output": 1.0
}
```

### 1.2. Predict with the Classification Model

**Endpoint:** `POST /api/solution/util/nova/ML/predict`

**Request Body:**
```json
{
    "id": "MyML-01",
    "data": {
        "X": { "source": [ [10, 15, 20] ] }
    }
}
```

**Response:**
The response includes the predicted probabilities for each class.

```json
{
    "status": true,
    "output": {
        "argmax": { "A": 0.98 },
        "argmin": { "C": 0.0 },
        "all": { "A": 0.98, "B": 0.02, "C": 0.0 }
    }
}
```

## 2. Regression (Linear Regression)

### 2.1. Train a Regression Model

**Endpoint:** `POST /api/solution/util/nova/ML/train`

**Request Body:**
```json
{
    "id": "Regression-01",
    "data": {
        "X": { "source": [[1], [2], [3], [4]] },
        "y": { "source": [2.0, 4.1, 6.0, 8.2], "y_type": "raw" }
    },
    "train": {
        "model": { "model_name": "LinearRegression" },
        "trainset": { "test_size": 0.25, "shuffle": true }
    }
}
```

**Response:**
The `output` field represents the Mean Squared Error (MSE) on the test set. A lower value indicates better performance.

```json
{
    "status": true,
    "output": 0.0
}
```

### 2.2. Predict with the Regression Model

**Endpoint:** `POST /api/solution/util/nova/ML/predict`

**Request Body:**
```json
{
    "id": "Regression-01",
    "data": { "X": { "source": [[5]] } }
}
```

**Response:**
The response provides the predicted value(s), their mean, and standard deviation.

```json
{
    "status": true,
    "output": {
        "predicted_values": [10.199999999999998],
        "mean": 10.199999999999998,
        "std": 0.0
    }
}
```

## 3. Clustering (K-Means)

### 3.1. Train a Clustering Model

Clustering is an unsupervised task, so only input data `X` is required.

**Endpoint:** `POST /api/solution/util/nova/ML/train`

**Request Body:**
```json
{
    "id": "Clustering-01",
    "data": { "X": { "source": [[1, 2], [1, 4], [5, 6], [6, 5]] } },
    "train": {
        "model": {
            "model_name": "KMeans",
            "model_params": { "n_clusters": 2, "random_state": 42 }
        }
    }
}
```

**Response:**
The `output` field returns a performance score for the clustering model (e.g., silhouette score).

```json
{
    "status": true,
    "output": 0.591051482474553
}
```

### 3.2. Predict with the Clustering Model

Predict the cluster assignment for new data points.

**Endpoint:** `POST /api/solution/util/nova/ML/predict`

**Request Body:**
```json
{
    "id": "Clustering-01",
    "data": { "X": { "source": [[2, 3], [5, 5]] } }
}
```

**Response:**
The response details the cluster assignments for the input data.
-   **`clusters`**: An array mapping each input data point to a cluster index.
-   **`unique_labels`**: A list of all unique cluster labels.
-   **`cluster_counts`**: A dictionary showing how many input points were assigned to each cluster.

```json
{
    "status": true,
    "output": {
        "clusters": [0, 1],
        "unique_labels": [0, 1],
        "cluster_counts": { "0": 1, "1": 1 }
    }
}
```