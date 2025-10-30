import uvicorn
from fastapi import FastAPI, Query, APIRouter, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import Response, RedirectResponse
import json

from API.AI.JSON_parser.Json_Parser import TrainJson_Parser, PredictJson_Parser

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
        
        self.app.include_router(self.app_router)
        
        
    def DL_Train(self, req:Request, jsonData = Body(...)):
        
        return self._success_output( TrainJson_Parser(jsonData).Start_Train() )
        
    
    def DL_Predict(self, req:Request, jsonData = Body(...)):
        
        return self._success_output( PredictJson_Parser(jsonData).Start_Prediction() )
    
    
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