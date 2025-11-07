import uvicorn
from fastapi import FastAPI, Query, APIRouter, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response, RedirectResponse
import json


# Deep Learning
from API.AI.JSON_parser.Json_Parser import TrainJson_Parser, PredictJson_Parser

# Machine Learning
from API.AI.JSON_parser.Json_Parser import MachineLearning_TrainJson_Parser, MachineLearning_PredictJson_Parser

class AI_API_SERVER():
    
    is_running:bool = False
    
    def __init__(
        self, 
        AI_Storage_Path = "./Storage", 
        server_ip:int = "0.0.0.0", 
        server_port:int =10302
    ):
        self.AI_Storage_Path = AI_Storage_Path
        
        self.server_ip = server_ip
        self.server_port = server_port
        
        self.is_running = False
        
        # app
        self.app = FastAPI()
        
        # 라우터
        self.app_router = APIRouter()
        self._add_router_url_paths()
        
        
        
    def _add_router_url_paths(self):
        
        # DL(딥러닝) 훈련
        self.app_router.post("/api/solution/util/nova/DL/train")(self.DL_Train)
        
        # DL(딥러닝) 예측
        self.app_router.post("/api/solution/util/nova/DL/predict")(self.DL_Predict)
        
        
        # ML(머신러닝) 훈련
        self.app_router.post("/api/solution/util/nova/ML/train")(self.ML_Train)
        
        # ML(머신러닝) 예측
        self.app_router.post("/api/solution/util/nova/ML/predict")(self.ML_Predict)
        
        self.app.include_router(self.app_router)
        
        
    def DL_Train(self, req:Request, jsonData = Body(...)):
        
        try:
            return self._success_output( TrainJson_Parser( self._output_jsonData( jsonData ) ).Start_Train() )
        except Exception as e:
            return self._failed_output( str(e) )
        
    
    def DL_Predict(self, req:Request, jsonData = Body(...)):
        
        try:
            return self._success_output( PredictJson_Parser( self._output_jsonData( jsonData ) ).Start_Prediction() )
        except Exception as e:
            return self._failed_output( str(e) )
    
    def ML_Train(self, req:Request, jsonData = Body(...)):
        
        try:
            return self._success_output( MachineLearning_TrainJson_Parser( self._output_jsonData( jsonData ) ).Start_Train() )
        except Exception as e:
            return self._failed_output( str(e) )
        
    
    def ML_Predict(self, req:Request, jsonData = Body(...)):
        
        try:
            return self._success_output( MachineLearning_PredictJson_Parser( self._output_jsonData( jsonData ) ).Start_Prediction() )
        except Exception as e:
            return self._failed_output( str(e) )
        
    
    def _output_jsonData(self, jsonData:any)->dict:
        if isinstance(jsonData, dict):
            return jsonData
        elif isinstance(jsonData, str):
            return json.loads(jsonData)
        elif isinstance(jsonData, bytes):
            return json.loads(jsonData)
    
    def _failed_output(self, reason:str)->dict:
        return {
            "status": False,
            "reason": reason
        }
    def _success_output(self, output_data:any)->dict:
        return {
            "status": True,
            "output": output_data
        }
    
    def Run(self)->bool:
        try:
            uvicorn.run(self.app, host=self.server_ip, port=self.server_port)
            self.is_running = True
            return True
        except:
            self.is_running = False
            return False