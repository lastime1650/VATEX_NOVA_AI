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

# WithId 관련
from API.AI.JSON_parser.Json_Parser import PushSampleStruct, WithId_AI_class

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
        
        self.WithId_AI_class = WithId_AI_class("./Storage/Withid")
        
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
        
        
        # New Function -> id식별 기반 데이터 HDD 축적과 훈련/예측 진행 (with_id)
        # AI 서버에서 샘플을 저장 및 관리가 가능하다.
        
        ## 샘플 (1개 또는 2개이상가능) HDD 저장 
        self.app_router.post("/api/solution/util/nova/with_id/sample/push")(self.WithId_Sample_Push)
        
        self.app_router.post("/api/solution/util/nova/with_id/sample/y/edit")(self.WithId_Sample_y_Edit)
        self.app_router.post("/api/solution/util/nova/with_id/sample/x/edit")(self.WithId_Sample_x_Edit)
        
        self.app_router.post("/api/solution/util/nova/with_id/sample/remove")(self.WithId_Sample_Remove)
        
        self.app_router.post("/api/solution/util/nova/with_id/ML/train")(self.WithId_Train_ML)
        self.app_router.post("/api/solution/util/nova/with_id/ML/predict")(self.WithId_Predict_ML)
        
        self.app_router.post("/api/solution/util/nova/with_id/DL/train")(self.WithId_Train_DL)
        self.app_router.post("/api/solution/util/nova/with_id/DL/predict")(self.WithId_Predict_DL)
        
        
        self.app.include_router(self.app_router)
    
    def WithId_Predict_DL(self, req:Request, jsonData = Body(...)):
        '''
        jsonData ->
        {
            "id": string,
            "sample": list ( X sample one )
        }
        '''
        try:
            parsed = self._output_jsonData( jsonData )
            
            
            id:str = parsed["id"]
            sample:list = parsed["sample"] # 1차원 샘플
                
            return self._success_output( self.WithId_AI_class.Sample_DL_Predict( id, sample ) )
        except Exception as e:
            return self._failed_output( str(e) )
    
    def WithId_Train_DL(self, req:Request, jsonData = Body(...)):
        try:
            parsed = self._output_jsonData( jsonData )
            
            
            id:str = parsed["id"]
            y_type:str = parsed["y_type"]
            train:dict = parsed["train"]
                
            return self._success_output( self.WithId_AI_class.Sample_DL_Train( id, y_type, train ) )
        except Exception as e:
            return self._failed_output( str(e) )
    
    def WithId_Predict_ML(self, req:Request, jsonData = Body(...)):
        '''
        jsonData ->
        {
            "id": string,
            "sample": list ( X sample one )
        }
        '''
        try:
            parsed = self._output_jsonData( jsonData )
            
            
            id:str = parsed["id"]
            sample:list = parsed["sample"] # 1차원 샘플
                
            return self._success_output( self.WithId_AI_class.Sample_ML_Predict( id, sample ) )
        except Exception as e:
            return self._failed_output( str(e) )
    
    def WithId_Train_ML(self, req:Request, jsonData = Body(...)):
        '''
        jsonData ->
        {
            "id": string,
            "y_type": string,
            "train": {
                "model": {
                    "model_name": e.g."RandomForestClassifier",
                    "model_params": e.g.{
                        "n_estimators": 100,
                        "random_state": 4
                    }
                },
                "trainset": e.g.{
                    "test_size": 0.25,
                    "shuffle": true,
                    "stratify": "y"
                }
            }
            
        }
        '''
        # HDD에 저장된 정보를 자체적으로 ML_Train() 요청 JSON 꼴로 만들어서 하는 전략이 제일 구현하기 편할 것이다.
        try:
            parsed = self._output_jsonData( jsonData )
            
            
            id:str = parsed["id"]
            y_type:str = parsed["y_type"]
            train:dict = parsed["train"]
                
            return self._success_output( self.WithId_AI_class.Sample_ML_Train( id, y_type, train  ) )
        except Exception as e:
            return self._failed_output( str(e) )
    
    def WithId_Sample_Remove(self, req:Request, jsonData = Body(...)):
        '''
        jsonData ->
        {
            "id": string,
            "sample_id": string
        }
        '''
        try:
            parsed = self._output_jsonData( jsonData )
            
            
            id:str = parsed["id"]
            sample_id:str = parsed["sample_id"]
                
            return self._success_output( self.WithId_AI_class.Sample_Remove( id, sample_id  ) )
        except Exception as e:
            return self._failed_output( str(e) )
    
    def WithId_Sample_x_Edit(self, req:Request, jsonData = Body(...)):
        '''
        jsonData ->
        {
            "id": string,
            "sample_id": string,
            "sample_x": ?
        }
        '''
        try:
            parsed = self._output_jsonData( jsonData )
            
            
            id:str = parsed["id"]
            sample_id:str = parsed["sample_id"]
            sample_x:list = parsed["sample_x"]
                
            return self._success_output( self.WithId_AI_class.Sample_x_Edit( id, sample_id, sample_x  ) )
        except Exception as e:
            return self._failed_output( str(e) )
        
    def WithId_Sample_y_Edit(self, req:Request, jsonData = Body(...)):
        '''
        jsonData ->
        {
            "id": string,
            "sample_id": string,
            "sample_y": ?
        }
        '''
        try:
            parsed = self._output_jsonData( jsonData )
            
            
            id:str = parsed["id"]
            sample_id:str = parsed["sample_id"]
            sample_y:any = parsed["sample_y"]
                
            return self._success_output( self.WithId_AI_class.Sample_y_Edit( id, sample_id, sample_y  ) )
        except Exception as e:
            return self._failed_output( str(e) )
    
    # 이벤트 축적
    def WithId_Sample_Push(self, req:Request, jsonData = Body(...)):
        '''
        jsonData ->
        {
            "id": string,
            "samples" : [
                {
                    "sample_id": string,
                    "sample_x": [], // features
                    "sample_y": ?
                }
            ]
        }
        '''
        try:
            parsed = self._output_jsonData( jsonData )
            
            
            id:str = parsed["id"]
            samples:list[PushSampleStruct] = []
            for sample in parsed["samples"]:
                samples.append( 
                    PushSampleStruct(
                        sample["sample_id"],
                        sample["sample_x"],
                        sample["sample_y"]
                    )
                )
                
            return self._success_output( self.WithId_AI_class.Sample_Push( id, samples  ) )
        except Exception as e:
            return self._failed_output( str(e) )
    
    
    ########################################################################################################
        
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