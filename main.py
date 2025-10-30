from API.APIServer import AI_API_SERVER

from API.AI.JSON_parser.Json_Parser import TrainJson_Parser, PredictJson_Parser


if __name__ == "__main__":
    
    AI_API_SERVER().Run()