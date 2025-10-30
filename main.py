from API.APIServer import AI_API_SERVER

from API.AI.JSON_parser.Json_Parser import TrainJson_Parser, PredictJson_Parser

if __name__ == "__main__":
    
    
    """obj = PredictJson_Parser(
        '''
        {
            "id": "classification-model-02",
            "data": {
                "X": {
                    "source": [
                        [11, 22, 33]
                    ]
                }
            }
        }
        '''
    )
    print( obj.Start_Prediction() )
    
    quit()"""

    """obj = TrainJson_Parser(
        '''
    {
        "id": "classification-model-02",
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
    }'''
    )
    print( obj.Start_Train() )"""
    
    sample = """{
            "id": "classification-model-01",
            "data": {
                "X": {
                    "source": [
                        [11, 22, 33]
                    ]
                }
            }
        }"""

    print(
        PredictJson_Parser(sample).Start_Prediction()
    )
    
    #API = AI_API_SERVER()
    #API.Run()