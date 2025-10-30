from abc import ABC, abstractmethod
import json
from enum import Enum
from typing import Optional

import tensorflow as tf
from keras import layers, Model, Input


# 훈련을 담당하는 로직

'''
    ================================================================================
    
    TrainLayer - Support 지원되는 Layer 오브젝트 클래스 정의
    
    ================================================================================
'''

## 훈련->레이어
class TrainHiddenLayerTypes(Enum):
    dense = 0
    lstm = 1
    drop_out = 2
    embedding = 3
    timedistributed = 4
    conv2d = 5      
    flatten = 6     
    
    
## 부모
class TrainHiddenLayerObject(ABC):
    
    @abstractmethod
    def Get_Layer(self)->any: # 자식은 해당 메서드를 필수로 구현해야한다.
        pass
    
    @abstractmethod
    def Get_Dict(self)->dict: # 자식은 해당 메서드를 필수로 구현해야한다.
        pass

## 자식 -> 일반(Dense) 은닉 층
class TrainHiddenDenseLayer(TrainHiddenLayerObject): 
    def __init__(self, layer:dict):
        # 활성화 함수
        self.activation_func:str = layer["activation_func"]
        # 뉴런개수 설정
        self.units:int = layer["units"]
    def Get_Layer(self)->any:
        return layers.Dense(
            units=self.units,
            activation=self.activation_func 
        )
    def Get_Dict(self)->dict:
        return {
            "type": TrainHiddenLayerTypes.dense.name,
            "activation_func": self.activation_func,
            "units": self.units
        }
        
## 자식 -> 드랍아웃(Dropout) 은닉 층
class TrainHiddenDropoutLayer(TrainHiddenLayerObject):
    def __init__(self, layer:dict):
        self.rate = layer["rate"]
    
    def Get_Layer(self)->any:
        return layers.Dropout(self.rate)
    
    def Get_Dict(self)->dict:
        return {
            "type": TrainHiddenLayerTypes.drop_out.name,
            "rate": self.rate
        }

## 자식 -> LSTM 은닉 층
class TrainHiddenLSTMLayer(TrainHiddenLayerObject):
    def __init__(self, layer:dict):
        self.units = layer["units"]
        self.return_sequences = layer.get("return_sequences", False)
    
    def Get_Layer(self):
        return layers.LSTM(
            units=self.units,
            return_sequences=self.return_sequences
        )
        
    def Get_Dict(self)->dict:
        return {
            "type": TrainHiddenLayerTypes.lstm.name,
            "units": self.units,
            "return_sequences": self.return_sequences
        }

## 자식 -> Embedding 층
class TrainHiddenEmbeddingLayer(TrainHiddenLayerObject):
    def __init__(self, layer: dict):
        self.input_dim = layer["input_dim"]        # 단어/카테고리 개수
        self.output_dim = layer["output_dim"]      # 임베딩 차원
        self.input_length = layer.get("input_length", None)  # 시퀀스 길이 (옵션)
    
    def Get_Layer(self):
        return layers.Embedding(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            input_length=self.input_length
        )
    
    def Get_Dict(self) -> dict:
        return {
            "type": TrainHiddenLayerTypes.embedding.name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_length": self.input_length
        }


## 자식 -> TimeDistributed 층
class TrainHiddenTimeDistributedLayer(TrainHiddenLayerObject):
    def __init__(self, layer: dict):
        self.units = layer["units"]
        self.activation_func = layer.get("activation_func", "linear")  # 기본 linear
    
    def Get_Layer(self):
        return layers.TimeDistributed(
            layers.Dense(units=self.units, activation=self.activation_func)
        )
    
    def Get_Dict(self) -> dict:
        return {
            "type": TrainHiddenLayerTypes.timedistributed.name,
            "units": self.units,
            "activation_func": self.activation_func
        }


## Conv2D 층
class TrainHiddenConv2DLayer(TrainHiddenLayerObject):
    def __init__(self, layer: dict):
        self.filters = layer["filters"]
        self.kernel_size = tuple(layer["kernel_size"]) # JSON에서는 리스트, Keras에서는 튜플
        self.activation_func = layer.get("activation_func", "relu") # 기본값 relu

    def Get_Layer(self) -> layers.Layer:
        return layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=self.activation_func
        )

    def Get_Dict(self) -> dict:
        return {
            "type": TrainHiddenLayerTypes.conv2d.name,
            "filters": self.filters,
            "kernel_size": list(self.kernel_size), # JSON으로 변환 시 다시 리스트로
            "activation_func": self.activation_func
        }

## Flatten 층
class TrainHiddenFlattenLayer(TrainHiddenLayerObject):
    def __init__(self, layer: dict):
        # Flatten 레이어는 별도의 파라미터가 필요 없음
        pass

    def Get_Layer(self) -> layers.Layer:
        return layers.Flatten()

    def Get_Dict(self) -> dict:
        return {
            "type": TrainHiddenLayerTypes.flatten.name
        }


# 헬퍼함수
def Helper__HiddensLayer_to_TrainHiddenLayerObject(Hiddens_value: list[dict]) -> list[TrainHiddenLayerObject]:
    output: list[TrainHiddenLayerObject] = []
    
    for Hidden_Layer in Hiddens_value:
        layer_type = str( Hidden_Layer["type"] ).lower()
        if layer_type == TrainHiddenLayerTypes.dense.name:
            output.append(TrainHiddenDenseLayer(Hidden_Layer))  
        elif layer_type == TrainHiddenLayerTypes.lstm.name:
            output.append(TrainHiddenLSTMLayer(Hidden_Layer))
        elif layer_type == TrainHiddenLayerTypes.drop_out.name:
            output.append(TrainHiddenDropoutLayer(Hidden_Layer))
        elif layer_type == TrainHiddenLayerTypes.embedding.name:
            output.append(TrainHiddenEmbeddingLayer(Hidden_Layer))
        elif layer_type == TrainHiddenLayerTypes.timedistributed.name:
            output.append(TrainHiddenTimeDistributedLayer(Hidden_Layer))
        elif layer_type == TrainHiddenLayerTypes.conv2d.name:
            output.append(TrainHiddenConv2DLayer(Hidden_Layer))
        elif layer_type == TrainHiddenLayerTypes.flatten.name:
            output.append(TrainHiddenFlattenLayer(Hidden_Layer))
    
    return output

'''
    ================================================================================
    
    TrainLayer 로직
    
    ================================================================================
'''

### Tensorflow Funtional API 기반으로 Layer를 쌓는다.
class TrainLayer():
    def __init__(self):
        self.InputLayer = None # Input 만 저장.
        self.Layer = None # 최신 레이어저장 (기대하는 것: Output이 최종 저장되어야함)   * 중요. Layer가 추가될때마다 덮어씌여지므로 요청시 실수가 없어야함.
        self.built_layers:list[dict] = [] # 기록된 Layer 정보
    
    # 입력층 추가
    def Add_Input_Layer(self, shape:list[int]):
        self.InputLayer = Input(shape=shape)
        self.Layer = self.InputLayer
        
        self.built_layers.append({"type": "input", "shape": shape}) # 기록
    
    # 은닉층 추가
    ## 단일
    def Add_Hidden_Layer(
        self,
        HiddenLayer:TrainHiddenLayerObject
    ):
        self.Layer = HiddenLayer.Get_Layer()(self.Layer) # 은닉층 축적
        self.built_layers.append(HiddenLayer.Get_Dict()) # 기록
    
    ## 복수
    def Add_Hidden_Layers(
        self,
        HiddenLayers:list[TrainHiddenLayerObject]
    ):
        for HiddenLayer in HiddenLayers:
            self.Layer = HiddenLayer.Get_Layer()(self.Layer) # 은닉층 축적
            self.built_layers.append(HiddenLayer.Get_Dict()) # 기록
    
    # 출력층 추가
    def Add_Output_Layer(self, units:int, activation_func:str):
        self.Layer = layers.Dense(units=units, activation=activation_func)(self.Layer) # 출력층 추가
        self.built_layers.append({"type":"output", "units":units ,"activation_func":activation_func}) # 기록
    
    # 모델 생성 (출력층까지 모두 성공적으로 축적되어야함)
    def Build_Model(self)->Model:
        if self.Layer is None or self.InputLayer is None:
            raise ValueError("입력층과 출력층이 정의되어야 모델을 구성할 수 있습니다.")
        
        return Model(inputs=self.InputLayer, outputs=self.Layer)
    
    # 지금까지 생성한 층에 대한 JSON 출력 (array)
    def ToDict(self)->dict:
        return {
            "layers": self.built_layers
        }