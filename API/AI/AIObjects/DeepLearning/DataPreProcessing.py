# API/AI/AIObjects/DataPreProcessing.py

from enum import Enum
import numpy as np
import os
import pickle

# [개선] scikit-learn의 강력한 전처리기 사용
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

class y_types(Enum):
    one_hot_encode = 0
    min_max = 1

class X_y_Train_Manager():
    def __init__(self, id: str, StoragePath: str, X: list, y: list, y_type: y_types):
        self.saved_dir = os.path.join(StoragePath, id)  # [개선] os.path.join으로 OS 호환성 확보
        self._make_dir() # 경로를 인자로 받을 필요 없음

        # 정규화된 데이터를 저장할 인스턴스 변수
        self.Normalized_X = self._set_X_normalizing(X)
        self.Normalized_y = self._set_y_normalized(y, y_type)

    def _make_dir(self):
        # [수정] self.saved_dir 자체를 생성해야 함
        os.makedirs(self.saved_dir, exist_ok=True)

    def _save_object(self, obj: any, filename: str):
        filepath = os.path.join(self.saved_dir, filename)
        with open(filepath, "wb") as f:
            # [수정] pickle.dump()를 사용하여 파일에 직접 저장
            pickle.dump(obj, f, protocol=5)
        print(f"  - Saved: {filename}")

    def _set_X_normalizing(self, Original_X: list) -> np.ndarray:
        X_np = np.array(Original_X)
        original_shape = X_np.shape
        
        # 3D 데이터도 처리 가능하도록 2D로 변환 후 스케일링
        X_reshaped = X_np.reshape(original_shape[0], -1)
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_reshaped)
        
        self._save_object(scaler, "x_scaler.pkl")
        
        # 원래 shape으로 복원
        return X_scaled.reshape(original_shape)

    def _set_y_normalized(self, Original_y: list, y_type: y_types) -> np.ndarray:
        y_np = np.array(Original_y)

        if y_type == y_types.min_max:
            scaler = MinMaxScaler()
            # fit_transform은 2D 배열을 기대하므로 reshape
            y_scaled = scaler.fit_transform(y_np.reshape(-1, 1))
            self._save_object(scaler, "y_scaler.pkl")
            return y_scaled
        
        elif y_type == y_types.one_hot_encode:
            # [개선] scikit-learn을 사용한 안정적이고 재사용 가능한 원-핫 인코딩
            # 1. 문자열 라벨("normal", "malware")을 정수(0, 1)로 변환
            label_encoder = LabelEncoder()
            y_int_encoded = label_encoder.fit_transform(y_np)
            self._save_object(label_encoder, "y_label_encoder.pkl") # 이 객체가 매핑 정보를 가짐
            
            # 2. 정수를 원-핫 벡터([1,0], [0,1])로 변환
            onehot_encoder = OneHotEncoder(sparse_output=False)
            y_one_hot = onehot_encoder.fit_transform(y_int_encoded.reshape(-1, 1))
            # onehot_encoder는 저장할 필요가 거의 없음 (label_encoder가 핵심)
            return y_one_hot

    # Getter 메서드 구현
    def Get_X(self) -> np.ndarray:
        return self.Normalized_X
        
    def Get_y(self) -> np.ndarray:
        return self.Normalized_y

class y_ObjectBase():
    def Convert_to_y(self, ModelPredictOutput:list[list])->dict:
        raise " NotImplementedError "
        
class y_scaler(y_ObjectBase):
    def __init__(self , Object:MinMaxScaler):
        self.Object = Object
    def Convert_to_y(self, ModelPredictOutput:list[list])->dict:
        '''
            ModelPredictOutput -> 
            [
                [ 0.123444, 1.0, 0.2424 ... ]
            ]
        '''
        '''
            Output (dict) ->
            {
                'output': 30.607603073120117
            }
            
        '''
        return {
            "output": float( ( list[list]( self.Object.inverse_transform(ModelPredictOutput) ) )[0][0] )
        }
        
        
class y_labeler(y_ObjectBase):
    def __init__(self , Object:LabelEncoder):
        self.Object = Object
    def Convert_to_y(self, ModelPredictOutput:list[list])->dict:
        '''
            ModelPredictOutput -> 
            [
                [ 0.123444, 1.0, 0.2424 ... ]
            ]
        '''
        '''
            Output (dict) ->
            {
                "argmax" : {        ->  Max한 값
                    "B" : 1.0
                }
                "argmin": {
                    "A" : 0.123444  ->  Min한 값
                }
                "all" : {           -> 전체 반환
                    "A" : 0.123444,
                    "B" : 1.0,
                    "C" : 0.2424
                }
            }
            
        '''
        Output:dict = {
            "argmax": {},
            "argmin": {},
            "all": {}
        }
        
        ModelPredict_to_1dim = ModelPredictOutput[0]
        
        classes:np.array = ( self.Object.classes_ )
        max_index_in_classes = np.argmax( ModelPredict_to_1dim )
        min_index_in_classes = np.argmin( ModelPredict_to_1dim )
        
        for i, class_lable in enumerate( classes ):
            if(max_index_in_classes == i):
                Output["argmax"] = {
                    str(class_lable) : float( ModelPredict_to_1dim[i] )
                }
                
            if (min_index_in_classes == i):
                Output["argmin"] = {
                    str(class_lable) : float( ModelPredict_to_1dim[i] )
                }
            
            Output["all"][str(class_lable)] = float( ModelPredict_to_1dim[i] )
        
        return Output

class X_y_Predict_Manager():
    def __init__(self, id: str, StoragePath: str):
        self.saved_dir = os.path.join(StoragePath, id)
        
        self.X_Scaler = self.load_file( self.saved_dir + "/x_scaler.pkl" )
        self.y_Object:y_ObjectBase = None
        
        # 미리 저장된 파일명에 따라 y_Object자식결정
        if( os.path.exists( os.path.join(self.saved_dir, "y_scaler.pkl") ) ):
            scalerObject = self.load_file( os.path.join(self.saved_dir, "y_scaler.pkl") )
            self.y_Object = y_scaler( scalerObject )
        elif ( os.path.exists( os.path.join(self.saved_dir, "y_label_encoder.pkl") ) ):
            labelObject = self.load_file( os.path.join(self.saved_dir, "y_label_encoder.pkl") )
            self.y_Object = y_labeler( labelObject )
        else:
            raise f"X_y_Predict_Manager() -> __init__ -> No yObject in {self.saved_dir} "
        
    def _make_dir(self):
        # [수정] self.saved_dir 자체를 생성해야 함
        os.makedirs(self.saved_dir, exist_ok=True)
        
    def load_file(self, filepath:str)->any:
        output = None
        with open(filepath, 'rb') as f:
            output = pickle.load(f)
        return output
    
    def Get_X_Scaler(self) -> MinMaxScaler:
        return self.X_Scaler
        
    def Get_y_Object(self) -> y_ObjectBase:
        return self.y_Object
            

class TrainData_PreProcessing():
    def __init__(self, id: str, StorageDir: str, data_key_value: dict):
        X_DataSource = data_key_value["X"]["source"]
        y_DataSource = data_key_value["y"]["source"]
        
        y_type_str = str(data_key_value["y"]["y_type"]).lower()
        if y_type_str == y_types.one_hot_encode.name:
            y_type = y_types.one_hot_encode
        elif y_type_str == y_types.min_max.name:
            y_type = y_types.min_max
        else:
            raise ValueError(f"Unsupported y_type: {y_type_str}")
        
        self.X_y_TrainManager = X_y_Train_Manager(
            id, StorageDir, X_DataSource, y_DataSource, y_type
        )