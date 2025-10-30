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

![initial](https://github.com/lastime1650/VATEX_NOVA_AI/blob/main/VATEX_NOVA_AI.png)

---

**NOVA** means "the NEW"
It provides the capability to train and predict frameworks on deep learning and AI very easily with only RestAPI(Json).

---

## Key Components

1. **Only RestAPI**
2. **Easy Connect to AI/DNN ! @Created Simple Framework for Easy**

---

## Examples

> ![note]
> 
> Python Example -> [https://github.com/lastime1650/VATEX_NOVA_AI/blob/main/api_request_client_sample.py](https://github.com/lastime1650/VATEX_NOVA_AI/blob/main/api_request_client_sample.py)
>

### Classification

### [1/2]. Train

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
          "source": [
            "A", "B", "C", "B", 
            "C", "A", "C", "B"
          ],
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
          "epochs": 100,
          "batch_size": 1,
          "validation_split": 0.25
      }
  }
}
```

This is JSON Request Body for the SoftMax-Classification


```python
[ ..., {'accuracy': 0.5, 'loss': 1.0380398035049438, 'val_accuracy': 0.0, 'val_loss': 1.1393353939056396}, {'accuracy': 0.5, 'loss': 1.0375632047653198, 'val_accuracy': 0.0, 'val_loss': 1.1382691860198975}, {'accuracy': 0.5, 'loss': 1.0382722616195679, 'val_accuracy': 0.0, 'val_loss': 1.1425538063049316}] # output
```

When the value is output, the value of the "accuracy metric" is included as many times as the number of training epochs.

<br>

### [2/2]. Prediction

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

When you request Predition, there only have "id", "X" in "data".

<br>

```json
{
  "argmax": {
    "A": 0.3658864200115204
  }, 
  "argmin": {
    "C": 0.2932397723197937
  }, 
  "all": {
    "A": 0.3658864200115204, 
    "B": 0.34087374806404114, 
    "C": 0.2932397723197937
  }
}
```

The output will provide the predicted data in the form of float. And Classification provides the two highest or lowest results, as well as the overall accuracy figures.


---



<br>


### Regration

### [1/2]. Train

```json
{
  "id": "regration-model-01",
  "data": {
      "X": {
          "source": [
              [10, 20, 30], [11, 22, 33], [90, 80, 70], [95, 88, 77],
              [50, 50, 50], [45, 55, 60], [12, 25, 35], [92, 81, 74]
          ]
      },
      "y": {
          "source": [24, 35, 3, 0, 35, 24, 64, 33],
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
          "epochs": 1000,
          "batch_size": 1,
          "validation_split": 0.25
      }
  }
}
```

This is JSON Request Body for the Sigmoid-Regression (mse)

<br>

```python
[ ..., {'loss': 0.023968348279595375, 'mae': 0.14522744715213776, 'val_loss': 0.20794758200645447, 'val_mae': 0.43592602014541626}, {'loss': 0.023929255083203316, 'mae': 0.14515821635723114, 'val_loss': 0.20819859206676483, 'val_mae': 0.4363044500350952}] # output
```

When the value is output, the value of the "mae metric" is included as many times as the number of training epochs.

### [2/2]. Prediction

```json
{
    "id": "regration-model-01",
    "data": {
        "X": {
            "source": [
                [11, 22, 33]
            ]
        }
    }
}
```

When you request Predition, there only have "id", "X" in "data".

<br>

```json
{"output": 28.43122673034668}
```

The output will provide the predicted data in the form of float.