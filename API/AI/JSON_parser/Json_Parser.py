# API/AI/JSON_parser/Json_Parser.py

from API.Timestamp import now_to_nano_int, nano_to_iso_string

import json
from typing import Dict, Optional
import API.AI.AIObjects.DeepLearning.LayerModel as Modeler
import API.AI.AIObjects.DeepLearning.DataPreProcessing as DPP
from keras.models import Model, load_model
import os, pickle
import numpy as np

class PredictJson_Parser():
    def __init__(self, PredictJson: dict, StoredDir: str = "./Storage"):
        self.PredictJson: dict = PredictJson
        self.id: str = self.PredictJson["id"]
        self.StoredDir = StoredDir
        self.model: Model = None
        self.history = None
        
        self.Pre__Pred_X:list = self.PredictJson["data"]["X"]["source"]
        
        self.X_y_Predict_Manager = DPP.X_y_Predict_Manager(self.id, self.StoredDir)
        self.X_scaler:DPP.MinMaxScaler = None
        self.y_object:DPP.y_ObjectBase = None
    
    def Start_Prediction(self)->dict:
        
        Pred_X:np.ndarray = None
        
        # 1. Model Load
        self._load_model()
        
        # 2. load X Scaler AND fit
        self._load_x_scaler()
        Pred_X = self.X_scaler.transform(self.Pre__Pred_X)
        # 3. load y Object ( scaler or labelEncoder for the trained one-hot-encode) AND fit
        self._load_y_object()
        
        
        # 4. Predict
        pred_result = self._predict( Pred_X )
        
        # 5. converting_by_yObject
        output = self.y_object.Convert_to_y(pred_result)
        
        # 6. output_Result
        return output
    
    
    def _load_model(self):
        self.model = load_model(os.path.join(self.StoredDir, self.id, "trained_model.keras"));
    def _predict(self, X:np.ndarray)->any:
        if(self.model):
            return self.model.predict(X)
        else:
            raise "None loaded Model"
    def _load_x_scaler(self):
        self.X_scaler = self.X_y_Predict_Manager.Get_X_Scaler()
    def _load_y_object(self):
        self.y_object = self.X_y_Predict_Manager.Get_y_Object()
    
    

class TrainJson_Parser():
    def __init__(self, TrainJson: dict, StoredDir: str = "./Storage"):
        self.TrainJson: dict = TrainJson
        self.id: str = self.TrainJson["id"]
        self.StoredDir = StoredDir
        self.model: Model = None
        self.history = None

    def Start_Train(self) -> list[dict]:
        print(f"--- Starting Training Pipeline for ID: {self.id} ---")
        
        # 1. 모델 생성 및 컴파일
        print("Step 1: Building and Compiling Model...")
        self._Build_Model()
        self._Model_compile()
        
        # 2. X, y 훈련 데이터 전처리
        print("\nStep 2: Preprocessing Data...")
        TrainDataManager = DPP.TrainData_PreProcessing(
            self.id,
            self.StoredDir,
            self.TrainJson["data"]
        )
        # 전처리된 데이터를 가져옴
        X = TrainDataManager.X_y_TrainManager.Get_X()
        y = TrainDataManager.X_y_TrainManager.Get_y()
        print(X)
        print(y)
        # 3. 모델 훈련
        print("\nStep 3: Fitting Model...")
        self._Model_fit(X, y)

        # 훈련된 모델 저장
        model_path = os.path.join(self.StoredDir, self.id, "trained_model.keras")
        self.model.save(model_path)
        print(f"\n--- Training Finished. Model saved to: {model_path} ---")

        # 훈련 결과를 반환
        
        output:list[dict] = []
        
        for metrics_key, v in dict(self.history.history).items():
           for i, metrics_value in enumerate( list[float](v) ):
                if(len(output) == 0 ):
                    # 리스트 초기화
                    output = [{} for _ in range(len(v))] # [{},{},,,,] 자리 준비
                
                output[i][metrics_key] = metrics_value
        print(output)
        
        
        return output

    def _Build_Model(self):
        Layer = self.TrainJson["train"]["layer"]
        TL = Modeler.TrainLayer()
        TL.Add_Input_Layer(shape=tuple(Layer["input"]["shape"])) # tuple로 변환 권장
        TL.Add_Hidden_Layers(Modeler.Helper__HiddensLayer_to_TrainHiddenLayerObject(Layer["hiddens"]))
        TL.Add_Output_Layer(Layer["output"]["units"], Layer["output"]["activation_func"])
        self.model = TL.Build_Model()
        self.model.summary()

    def _Model_compile(self):
        if not self.model:
            raise RuntimeError("Model has not been built yet. Cannot compile.")
        Compile = self.TrainJson["train"]["compile"]
        self.model.compile(
            optimizer=Compile["optimizer_func"],
            loss=Compile["loss_func"],
            metrics=Compile["metrics"]  # [수정] list[str]()는 유효하지 않음
        )
        print("  - Model Compiled Successfully.")

    def _Model_fit(self, X: any, y: any):
        if not self.model:
            raise RuntimeError("Model has not been compiled yet. Cannot fit.")
        Fit = self.TrainJson["train"]["fit"]
        self.history = self.model.fit(
            x=X,
            y=y,
            batch_size=Fit["batch_size"],
            epochs=Fit["epochs"],
            validation_split=float(Fit.get("validation_split", 0.0)) # .get()으로 안정성 확보
        )
        

from API.AI.AIObjects.MachineLearning.Trainer import  MachineLearning_Enum, helper_name_to_model_enum, helper_name_to_ML_y_types, ML_y_types, Make_Train_Test_sets, Model_fit, Make_Clustering_sets

class MachineLearning_TrainJson_Parser():
    def __init__(self, TrainJson: dict, StoredDir: str = "./Storage"):
        self.TrainJson: dict = TrainJson
        self.id: str = self.TrainJson["id"]
        self.StoredDir = StoredDir
        
        '''
        {
            
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
                    "y_type": "label"(라벨링 데이터) 또는 "raw"(그대로 Train_y로 사용)
                }
            },
            
            "train": {
                
                "model":{
                    "model_name": "RandomForestClassifier",                     # 필수
                    "model_params" : { ... } -> **params 로 그대로 제공될 예정      # 필수 ( 선정된 모델에 따라 파라미터 설정(인자명또한 매치해야함) )
                }
                
                "trainset":{
                    
                    # 만약 필요없는 key-value의 경우는 삽입 안해도된다. 
                    
                    test_size=0.2,
                    train_size=None,
                    random_state=None,
                    shuffle=True,
                    stratify=None,
                }
                
            }
        }
        '''
        
        #X,y
        X = self.TrainJson["data"]["X"]["source"]
        y = None
        
        self.X_train = None; self.X_test = None; self.y_train = None; self.y_test = None
        
        self.train_model_name:str = self.TrainJson["train"]["model"]["model_name"]
        self.train_model_params:dict = dict( self.TrainJson["train"]["model"] ).get("model_params")
        print(self.train_model_params)
        
        # Model Enum
        self.ModelEnum = helper_name_to_model_enum(self.train_model_name)
        
        if self.TrainJson["data"].get("y") and self.TrainJson["data"]["y"].get("source") and self.TrainJson["data"]["y"]["source"] != None:
            y = self.TrainJson["data"]["y"]["source"]
            
            y_type:ML_y_types = helper_name_to_ML_y_types( self.TrainJson["data"]["y"]["y_type"] )
        
            TrainSet = self._check_trainset_dict(self.TrainJson["train"]["trainset"], y)
            
            self.X_train, self.X_test, self.y_train, self.y_test = Make_Train_Test_sets(self.id, self.StoredDir, self.ModelEnum, X,y, y_type, **(TrainSet) )
        else:
            # if Clustering? 
            self.X_train = Make_Clustering_sets(self.id, self.StoredDir, X)
        
        
        
    def Start_Train(self) -> float: # 정확도 값
        
        return Model_fit(
            self.id, self.StoredDir,self.ModelEnum,
            self.X_train, self.X_test, self.y_train, self.y_test, 
            **self.train_model_params if self.train_model_params else {}
        )

        
    def _check_trainset_dict(self, trainset_Dict: dict, y: list) -> dict:
        """
        trainset_Dict 내에서 'stratify' 값이 문자열 'y'이면 실제 y 리스트로 치환.
        나머지는 그대로 반환.
        """
        if not isinstance(trainset_Dict, dict):
            raise ValueError("trainset_Dict는 dict 형태여야 합니다.")
        
        # 복사해서 원본 손상 방지
        checked_dict = trainset_Dict.copy()
        
        # stratify 값이 문자열 "y"이면 실제 y로 변경
        if "stratify" in checked_dict and checked_dict["stratify"] == "y":
            checked_dict["stratify"] = y
        else:
            # stratify 키가 없거나 "y"가 아니면 그대로 둔다.
            pass
        
        return checked_dict
    

from API.AI.AIObjects.MachineLearning.Trainer import ML_Predict
    
class MachineLearning_PredictJson_Parser():
    def __init__(self, TrainJson: dict, StoredDir: str = "./Storage"):
        self.TrainJson: dict = TrainJson
        self.id: str = self.TrainJson["id"]
        self.StoredDir = StoredDir
        '''
        {
            
            "data": {
                "X": {
                    "source": [
                        [10, 20, 30]
                    ]
                }
            }
        }
        '''
        self.X = self.TrainJson["data"]["X"]["source"]
        
        
    def Start_Prediction(self):
        return ML_Predict(self.id, self.StoredDir, self.X)
    


class PushSampleStruct():
    def __init__(self, sample_id:str, sample_x:list, sample_y:any):
        self.sample_id = sample_id
        self.sample_x = sample_x
        self.sample_y = sample_y
    def Output_in_dict(self):
        return {
            "sample_id" : self.sample_id,
            "sample_x" : self.sample_x,
            "sample_y" : self.sample_y
        }

class WithIdAIStatusManager():
    def __init__(self, data:dict):
        '''
        {
            "samples_x_count" : 1234,
            "train_history" : [
                {
                    "nano_timestamp": <unsigned long long>,
                    "nano_timestamp_iso8901" : string,
                    "at_samples_x_count" : <unsigned long long>, // 당시 훈련시, 했던 샘플 개수
                    "train_result": {?} // 요청값에 따라 다름
                },,,
            ],
            "predict_history": [
                {
                    "nano_timestamp": <unsigned long long>,
                    "nano_timestamp_iso8901" : string,
                    "at_samples_x_count" : <unsigned long long>, // 당시 예측시, 했던 샘플 개수
                    "predict_result": {?} //
                }
            ]
        }
        '''
        self.id = data["id"]
        self.samples = data["samples"]
        self.status = data["status"]
        
        self._is_updated = False
        
    def Append_train_history(self, nano_timestamp:int, train_result:dict, with_update_at_end:bool = False, with_return:bool = False):
        self.status["train_history"].append(
            {
                "nano_timestamp": nano_timestamp,
                "nano_timestamp_iso8901" : nano_to_iso_string(nano_timestamp),
                "at_samples_x_count" : len(self.samples),
                "train_result": train_result
            }
        )
        if(with_update_at_end and self._is_updated == False):
            self.Update()
        if(with_return):
            return self.status
        
    def Append_predict_history(self, nano_timestamp:int, predict_result:dict, with_update_at_end:bool = False, with_return:bool = False):
        self.status["predict_history"].append(
            {
                "nano_timestamp": nano_timestamp,
                "nano_timestamp_iso8901" : nano_to_iso_string(nano_timestamp),
                "at_samples_x_count" : len(self.samples),
                "predict_result": predict_result
            }
        )
        if(with_update_at_end and self._is_updated == False):
            self.Update()
        if(with_return):
            return self.status
        
    def Update(self):
        # status 업데이트
        self.status["samples_x_count"] = len(self.samples)
        
        self._is_updated = True
        
    def OutputStatus(self, is_with_update:bool = False)->dict:
        
        if(is_with_update and self._is_updated == False ):
            self.Update()
        
        return self.status
        
        
    
class WithId_AI_class():
    def __init__(self, StoragePath:str):
        self.StoragePath = StoragePath
    def _make_dir(self, dir:str):
        os.makedirs(dir, exist_ok=True)
    def _get_filepath(self, id:str)->str:
        saved_dir = os.path.join(self.StoragePath, id) # StoragePath/{id} <dir>
        self._make_dir(saved_dir) # Creating Dir
        
        SamplesFileName = f"{id}_samples"
        
        # 파일 체크.
        filepath  = os.path.join(saved_dir, SamplesFileName)
        return filepath
    def _get_samplefile_by_hdd(self, filepath:str)->dict:
        data = {}
        with open( filepath, "rb" ) as f:
            data = pickle.load(f)
        return data
    def _set_samplefile_to_hdd(self, filepath:str, InputData:dict):
        with open( filepath, "wb" ) as f:
            pickle.dump(InputData, f, protocol=5)
    
        
    def Sample_Push(self, id:str, samples:list[PushSampleStruct]):
        # 파일 체크.
        filepath  = self._get_filepath(id)
        if os.path.exists( filepath ):
            # case: 파일 있음
            
            # 기존 파일 열고 -> 이어저장
            data = self._get_samplefile_by_hdd(filepath)
            if( data["id"] != id ):
                raise f"{filepath}에 저장된 id와 다름."
            
            #data["samples"] += [ sample.Output_in_dict() for sample in samples ]
            
            new = [ sample.Output_in_dict() for sample in samples ]
            for sample in ( new ):
                
                # 샘플 id 충돌확인 -> 일치(충돌)시 덮어쓰기 진행
                is_matched = False
                for ii, saved_sample in enumerate(data["samples"]):
                    if saved_sample["sample_id"] == sample["sample_id"]:
                        data["samples"][ii] = sample
                        is_matched = True
                        break
                    
                # 충돌아니었을 때 Append 
                if is_matched == False:
                    data["samples"].append(sample)
                        
            
            # Status 업데이트
            data["status"] = WithIdAIStatusManager(data).OutputStatus(True)
            #print(data["status"])
            
            # 이어저장
            self._set_samplefile_to_hdd(filepath, data)
                
        else:
            # case: 파일 없음
            data = {
                    "id" : id,
                    "samples" : [ sample.Output_in_dict() for sample in samples ],
                    "status":{
                        "samples_x_count" : 1,
                        "train_history": [],
                        "predict_history": []
                    }
                }
            
            # 새로 쓰기
            self._set_samplefile_to_hdd(filepath, data)
            
        return True
    
    def Sample_x_Edit(self, id:str, sample_id:str, sample_x:list):
        filepath  = self._get_filepath(id)
        if os.path.exists( filepath ):
            data = self._get_samplefile_by_hdd(filepath)
            for sample in data["samples"]:
                if sample["sample_id"] == sample_id:
                    # matched
                    sample["sample_x"] = sample_x
                    self._set_samplefile_to_hdd(filepath, data)
                    print(data)
                    return True
            raise "No Matched sample_id"
        else:
            raise "No File exists"
    def Sample_y_Edit(self, id:str, sample_id:str, sample_y:any)->bool:
        filepath  = self._get_filepath(id)
        if os.path.exists( filepath ):
            data = self._get_samplefile_by_hdd(filepath)
            for sample in data["samples"]:
                if sample["sample_id"] == sample_id:
                    # matched
                    sample["sample_y"] = sample_y
                    self._set_samplefile_to_hdd(filepath, data)
                    print(data)
                    return True
            raise "No Matched sample_id"
        else:
            raise "No File exists"
        
    def Sample_Remove(self, id:str, sample_id:str):
        filepath  = self._get_filepath(id)
        if os.path.exists( filepath ):
            data = self._get_samplefile_by_hdd(filepath)
            samples:list = data["samples"]
            
            # 역순으로 제거
            for i in reversed(range(len(samples))):
                if samples[i]["sample_id"] == sample_id :
                    samples.pop(i)
                    
                    
            # Status 업데이트
            data["status"] = WithIdAIStatusManager(data).OutputStatus(True)
            #print(data["status"])
            
            self._set_samplefile_to_hdd(filepath, data)
            return True
        else:
            raise "No File exists"
        
    # MachineLearning - Train
    def Sample_ML_Train(self, id:str, y_type:str, train:dict):
        
        
        # HDD로부터 확인
        filepath  = self._get_filepath(id)
        data = {}
        if os.path.exists( filepath ):
            data = self._get_samplefile_by_hdd(filepath)
        else:
            raise "No Samples File"
        
        if data["id"] != id:
            raise "MisMatch id"
        
        # ML :: Train :: Sample 만들기
        data_X_sources:list = []
        data_y_sources:list = []
        for source in data["samples"]:
            # X,y 서로간 index는 Matching 되어야한다. 무조건!!@(*^$#*&#@$^#*^%)
            data_X_sources.append( source["sample_x"] )
            
            if( source["sample_y"] == None or ( dict(source).get("sample_y") == None) ):
                continue
            else:
                data_y_sources.append( source["sample_y"] )
                
        if( len(data_y_sources) == 0 ):
            data_y_sources = None
            
        
        print(
            {
                "id" : id,
                "data": {
                    "X": {
                        "source": data_X_sources # 2차원
                    },
                    "y": {
                        "source": data_y_sources, # 2차원 ( None/null 이면 클러스터링 취급)
                        "y_type": y_type
                    }
                },
                "train": train
            }
        )
        
        ML_TRAIN_JSON = \
        {
            "id" : id,
            "data": {
                "X": {
                    "source": data_X_sources # 2차원
                },
                "y": {
                    "source": data_y_sources, # 2차원 ( None/null 이면 클러스터링 취급)
                    "y_type": y_type
                }
            },
            "train": train
        }
        
        
        output_train = 0.0
        try:
            output_train = MachineLearning_TrainJson_Parser(
                ML_TRAIN_JSON
            ).Start_Train()
            
            # Status 업데이트
            data["status"] = WithIdAIStatusManager(data).Append_train_history(now_to_nano_int(), {"type": "ML", "output": output_train},True, True )
            #print(data["status"])
            self._set_samplefile_to_hdd(filepath, data)
            
            return output_train
        except Exception as e:
            return e
        
        
        
        
    # MachineLearning - Predict
    def Sample_ML_Predict(self, id:str, X:list):
        filepath  = self._get_filepath(id)
        data = {}
        if os.path.exists( filepath ):
            data = self._get_samplefile_by_hdd(filepath)
        else:
            raise "No Samples File"
        
        output_predict:dict = {}
        try:
            output_predict = MachineLearning_PredictJson_Parser(
                {
                    "id" : id,
                    "data": {
                        "X":{
                            "source": [X] # to 2D
                        }
                    }
                }
            ).Start_Prediction()
            
            # Status 업데이트
            data["status"] = WithIdAIStatusManager(data).Append_predict_history(now_to_nano_int(), {"type": "ML", "output": output_predict},True, True )
            #print(data["status"])
            self._set_samplefile_to_hdd(filepath, data)
            
            return output_predict
        except Exception as e:
            return e
        
    # DeepLearning - Train
    def Sample_DL_Train(self, id:str, y_type:str, train:dict):
        
        # HDD로부터 확인
        filepath  = self._get_filepath(id)
        data = {}
        if os.path.exists( filepath ):
            data = self._get_samplefile_by_hdd(filepath)
        else:
            raise "No Samples File"
        
        if data["id"] != id:
            raise "MisMatch id"
        
        # ML :: Train :: Sample 만들기
        data_X_sources:list = []
        data_y_sources:list = []
        for source in data["samples"]:
            # X,y 서로간 index는 Matching 되어야한다. 무조건!!@(*^$#*&#@$^#*^%)
            data_X_sources.append( source["sample_x"] )
            data_y_sources.append( source["sample_y"] )
        
        DL_TRAIN_JSON = \
        {
            "id": id,
            "data": {
                "X": {
                    "source": data_X_sources # 2차원
                },
                "y": {
                    "source": data_y_sources, # 2차원 ( None/null 이면 클러스터링 취급)
                    "y_type": y_type
                }
            },
            "train": train
        }
        
        output_train:list[dict] = []
        try:
            output_train = TrainJson_Parser(
                DL_TRAIN_JSON
            ).Start_Train()
            
            # Status 업데이트
            data["status"] = WithIdAIStatusManager(data).Append_train_history(now_to_nano_int(), {"type": "DL", "output": output_train},True, True )
            #print(data["status"])
            self._set_samplefile_to_hdd(filepath, data)
            
            return output_train
        except Exception as e:
            return e
        
    # DeepLearning - Predict
    def Sample_DL_Predict(self, id:str, X:list):
        
        filepath  = self._get_filepath(id)
        data = {}
        if os.path.exists( filepath ):
            data = self._get_samplefile_by_hdd(filepath)
        else:
            raise "No Samples File"
        
        output_predict:dict = {}
        try:
            output_predict = PredictJson_Parser(
            {
                "id" : id,
                "data": {
                    "X":{
                        "source": [X] # to 2D
                    }
                }
            }
        ).Start_Prediction()
            
            # Status 업데이트
            data["status"] = WithIdAIStatusManager(data).Append_predict_history(now_to_nano_int(), {"type": "DL", "output": output_predict},True, True )
            #print(data["status"])
            self._set_samplefile_to_hdd(filepath, data)
            
            return output_predict
        except Exception as e:
            return e
        
    def Get_Status(self, id:str):
        filepath  = self._get_filepath(id)
        data = {}
        if os.path.exists( filepath ):
            data = self._get_samplefile_by_hdd(filepath)
            
            return WithIdAIStatusManager(data).OutputStatus()
        else:
            raise "No Samples File"
        
        