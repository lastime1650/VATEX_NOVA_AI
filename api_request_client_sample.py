import requests, json
'''
    Let me show you how to request to VATEX_NOVA_AI-API_Server !
'''

# ip/port
API_SERVER_IP = "192.168.1.205"
API_SERVER_PORT = 10302

# path
Deep_Learning_Train_Path = "/api/solution/util/nova/DL/train"
Deep_Learning_Prediction_Path = "/api/solution/util/nova/DL/predict"

# Scenario 1. Classification
Scenario_1_Train_Req = \
"""
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
          "epochs": 10,
          "batch_size": 1,
          "validation_split": 0.25
      }
  }
}
"""
Scenario_1_Prediction_Req = \
"""
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
"""

# request for training
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+Deep_Learning_Train_Path, data=json.dumps(Scenario_1_Train_Req)).content )
'''
    would be outputing this
    ->
    
    b'{"status":true,"output":[{"accuracy":0.3333333432674408,"loss":1.0891636610031128,"val_accuracy":0.5,"val_loss":0.9032398462295532},
    {"accuracy":0.3333333432674408,"loss":1.082886815071106,"val_accuracy":0.5,"val_loss":0.9078031778335571},{"accuracy":0.5,"loss":1.0796005725860596,
    "val_accuracy":0.5,"val_loss":0.9137726426124573},{"accuracy":0.5,"loss":1.0730088949203491,"val_accuracy":0.5,"val_loss":0.9171659350395203},
    {"accuracy":0.5,"loss":1.071439266204834,"val_accuracy":0.5,"val_loss":0.9234049916267395},{"accuracy":0.5,"loss":1.0683495998382568,"val_accuracy":0.5,"val_loss":0.9279903173446655},
    {"accuracy":0.5,"loss":1.0659137964248657,"val_accuracy":0.5,"val_loss":0.9341307878494263},{"accuracy":0.5,"loss":1.0645748376846313,"val_accuracy":0.5,"val_loss":0.9388964176177979},
    {"accuracy":0.5,"loss":1.0608512163162231,"val_accuracy":0.5,"val_loss":0.9400978684425354},{"accuracy":0.5,"loss":1.059830904006958,"val_accuracy":0.5,"val_loss":0.94476318359375}]}'
'''

# lets go prediction
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+Deep_Learning_Prediction_Path, data=json.dumps(Scenario_1_Prediction_Req)).content )
'''
    would be outputing this
    ->
    
    b'{"status":true,"output":{"argmax":{"B":0.33860063552856445},"argmin":{"A":0.32945770025253296},"all":{"A":0.32945770025253296,"B":0.33860063552856445,"C":0.3319416642189026}}}'
'''