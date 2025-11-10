import requests, json

API_SERVER_IP = "192.168.1.205"
API_SERVER_PORT = 10302

# path
WithId_Samples_Push_PATH = "/api/solution/util/nova/with_id/sample/push"
WithId_Samples_y_Edit_PATH = "/api/solution/util/nova/with_id/sample/y/edit"
WithId_Samples_x_Edit_PATH = "/api/solution/util/nova/with_id/sample/x/edit"
WithId_Sample_Remove_PATH = "/api/solution/util/nova/with_id/sample/remove"
WithId_Sample_ML_Train_PATH = "/api/solution/util/nova/with_id/ML/train"
WithId_Sample_ML_Predict_PATH = "/api/solution/util/nova/with_id/ML/predict"
WithId_Sample_DL_Train_PATH = "/api/solution/util/nova/with_id/DL/train"
WithId_Sample_DL_Predict_PATH = "/api/solution/util/nova/with_id/DL/predict"
WithId_Status_PATH = "/api/solution/util/nova/with_id/status"

# Event Push [1/4]
Sample_Push = \
"""
{
    "id": "MySample-1",
    "samples": [
        {
            "sample_id": "test0",
            "sample_x": [1,2,3],
            "sample_y": "A"
        }
    ]
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Samples_Push_PATH, data=json.dumps(Sample_Push)).content )

# Event Push [2/4]
Sample_Push = \
"""
{
    "id": "MySample-1",
    "samples": [
        {
            "sample_id": "test1",
            "sample_x": [3,2,1],
            "sample_y": "A"
        }
    ]
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Samples_Push_PATH, data=json.dumps(Sample_Push)).content )

# Event Push [3/4]
Sample_Push = \
"""
{
    "id": "MySample-1",
    "samples": [
        {
            "sample_id": "test2",
            "sample_x": [9,9,9],
            "sample_y": "B"
        }
    ]
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Samples_Push_PATH, data=json.dumps(Sample_Push)).content )

# Event Push [4/4]
Sample_Push = \
"""
{
    "id": "MySample-1",
    "samples": [
        {
            "sample_id": "test3",
            "sample_x": [8,8,8],
            "sample_y": "B"
        }
    ]
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Samples_Push_PATH, data=json.dumps(Sample_Push)).content )

'''
    => 4 times Push result
    
    b'{"status":true,"output":true}'
    b'{"status":true,"output":true}'
    b'{"status":true,"output":true}'
    b'{"status":true,"output":true}'
    
'''


'''
Sample_y_Edit = \
"""
{
    "id": "MySample-1",
    "sample_id": "test0",
    "sample_y": "B"
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Samples_y_Edit_PATH, data=json.dumps(Sample_y_Edit)).content )

Sample_x_Edit = \
"""
{
    "id": "MySample-1",
    "sample_id": "test0",
    "sample_x": [0,0,0]
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Samples_x_Edit_PATH, data=json.dumps(Sample_x_Edit)).content )

WithId_Sample_Remove = \
"""
{
    "id": "MySample-1",
    "sample_id": "test0"
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Sample_Remove_PATH, data=json.dumps(WithId_Sample_Remove)).content )
'''

# ML - Train TEST
Sample_ML_TRAIN = \
"""
{
    "id": "MySample-1",
    "y_type": "label",
    "train": {
        "model": {
            "model_name": "RandomForestClassifier",
            "model_params": {
                "n_estimators": 100,
                "random_state": 4
            }
        },
        "trainset": {
            "test_size": 0.25
        }
    }
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Sample_ML_Train_PATH, data=json.dumps(Sample_ML_TRAIN)).content )
'''
    => Result
    
    b'{"status":true,"output":1.0}' #  accuracy tested float 
'''

# ML - Predict TEST
Sample_ML_PREDICT = \
"""
{
    "id": "MySample-1",
    "sample": [3,3,3]
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Sample_ML_Predict_PATH, data=json.dumps(Sample_ML_PREDICT)).content )
'''
    => Result
    
    b'{"status":true,"output":{"argmax":{"A":0.62},"argmin":{"B":0.38},"all":{"A":0.62,"B":0.38}}}'
'''

# (Binary_crossentropy)
# DL - Train 
Sample_DL_TRAIN = \
"""
{
    "id": "MySample-1",
    "y_type": "binary",
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
          "loss_func": "binary_crossentropy",
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
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Sample_DL_Train_PATH, data=json.dumps(Sample_DL_TRAIN)).content )
'''
    => Result
    
    b'{"status":true,"output":[ .. ,{"accuracy":0.3333333432674408,"loss":0.5691565871238708,"val_accuracy":1.0,"val_loss":0.2815495431423187},{"accuracy":0.3333333432674408,"loss":0.5659993290901184,"val_accuracy":1.0,"val_loss":0.2793162763118744}]}'
'''


# DL - Predict
Sample_DL_PREDICT = \
"""
{
    "id": "MySample-1",
    "sample": [3,3,3]
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Sample_DL_Predict_PATH, data=json.dumps(Sample_DL_PREDICT)).content )
'''
    => Result
    
    b'{"status":true,"output":{"argmax":{"B":0.5707857608795166},"argmin":{"A":0.4292142391204834},"all":{"A":0.4292142391204834,"B":0.5707857608795166}}}'
'''


WithId_STATUS = \
"""
{
    "id": "MySample-1"
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Status_PATH, data=json.dumps(WithId_STATUS)).content )

'''
Output -> 
{
  "samples_x_count": 4,
  "train_history": [
    {
      "nano_timestamp": 1762759823874624737,
      "nano_timestamp_iso8901": "2025-11-10T16:30:23.874624737Z",
      "at_samples_x_count": 4,
      "train_result": {
        "type": "ML",
        "output": 1.0
      }
    },
    {
      "nano_timestamp": 1762759825539053266,
      "nano_timestamp_iso8901": "2025-11-10T16:30:25.539053266Z",
      "at_samples_x_count": 4,
      "train_result": {
        "type": "DL",
        "output": [
          { "accuracy": 0.6667, "loss": 0.6890, "val_accuracy": 1.0, "val_loss": 0.6730 },
          { "accuracy": 0.6667, "loss": 0.6861, "val_accuracy": 1.0, "val_loss": 0.6688 },
          { "accuracy": 0.6667, "loss": 0.6836, "val_accuracy": 1.0, "val_loss": 0.6644 },
          { "accuracy": 0.6667, "loss": 0.6809, "val_accuracy": 1.0, "val_loss": 0.6607 },
          { "accuracy": 0.6667, "loss": 0.6795, "val_accuracy": 1.0, "val_loss": 0.6551 },
          { "accuracy": 0.6667, "loss": 0.6761, "val_accuracy": 1.0, "val_loss": 0.6515 },
          { "accuracy": 0.6667, "loss": 0.6742, "val_accuracy": 1.0, "val_loss": 0.6467 },
          { "accuracy": 0.6667, "loss": 0.6717, "val_accuracy": 1.0, "val_loss": 0.6420 },
          { "accuracy": 0.6667, "loss": 0.6698, "val_accuracy": 1.0, "val_loss": 0.6368 },
          { "accuracy": 0.6667, "loss": 0.6673, "val_accuracy": 1.0, "val_loss": 0.6319 }
        ]
      }
    }
  ],
  "predict_history": [
    {
      "nano_timestamp": 1762759823887863789,
      "nano_timestamp_iso8901": "2025-11-10T16:30:23.887863789Z",
      "at_samples_x_count": 4,
      "predict_result": {
        "type": "ML",
        "output": {
          "argmax": { "A": 0.64 },
          "argmin": { "B": 0.36 },
          "all": { "A": 0.64, "B": 0.36 }
        }
      }
    },
    {
      "nano_timestamp": 1762759825665179864,
      "nano_timestamp_iso8901": "2025-11-10T16:30:25.665179864Z",
      "at_samples_x_count": 4,
      "predict_result": {
        "type": "DL",
        "output": {
          "argmax": { "B": 0.5011 },
          "argmin": { "A": 0.4989 },
          "all": { "A": 0.4989, "B": 0.5011 }
        }
      }
    }
  ]
}
'''