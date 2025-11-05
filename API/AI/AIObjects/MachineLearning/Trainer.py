from typing import Optional
from typing import Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

from enum import Enum
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN

from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error, silhouette_score
import numpy as np

import os

class MachineLearning_Enum(Enum):
    # 분류(Classification)
    LOGISTIC_REGRESSION = "LogisticRegression"
    KNN_CLASSIFIER = "KNeighborsClassifier"
    SVM_CLASSIFIER = "SVC"
    DECISION_TREE_CLASSIFIER = "DecisionTreeClassifier"
    RANDOM_FOREST_CLASSIFIER = "RandomForestClassifier"
    GRADIENT_BOOSTING_CLASSIFIER = "GradientBoostingClassifier"
    NAIVE_BAYES = "GaussianNB"

    # 회귀(Regression)
    LINEAR_REGRESSION = "LinearRegression"
    KNN_REGRESSOR = "KNeighborsRegressor"
    SVR = "SVR"
    DECISION_TREE_REGRESSOR = "DecisionTreeRegressor"
    RANDOM_FOREST_REGRESSOR = "RandomForestRegressor"
    GRADIENT_BOOSTING_REGRESSOR = "GradientBoostingRegressor"

    # 군집화(Clustering)
    KMEANS = "KMeans"
    DBSCAN = "DBSCAN"

    def create_model(self, *args, **kwargs)->any:
        """Enum 값에 해당하는 실제 사이킷런 모델 객체를 생성"""
        if self == MachineLearning_Enum.LOGISTIC_REGRESSION:
            return LogisticRegression(*args, **kwargs)
        elif self == MachineLearning_Enum.KNN_CLASSIFIER:
            return KNeighborsClassifier(*args, **kwargs)
        elif self == MachineLearning_Enum.SVM_CLASSIFIER:
            return SVC(*args, **kwargs)
        elif self == MachineLearning_Enum.DECISION_TREE_CLASSIFIER:
            return DecisionTreeClassifier(*args, **kwargs)
        elif self == MachineLearning_Enum.RANDOM_FOREST_CLASSIFIER:
            return RandomForestClassifier(*args, **kwargs)
        elif self == MachineLearning_Enum.GRADIENT_BOOSTING_CLASSIFIER:
            return GradientBoostingClassifier(*args, **kwargs)
        elif self == MachineLearning_Enum.NAIVE_BAYES:
            return GaussianNB(*args, **kwargs)

        elif self == MachineLearning_Enum.LINEAR_REGRESSION:
            return LinearRegression(*args, **kwargs)
        elif self == MachineLearning_Enum.KNN_REGRESSOR:
            return KNeighborsRegressor(*args, **kwargs)
        elif self == MachineLearning_Enum.SVR:
            return SVR(*args, **kwargs)
        elif self == MachineLearning_Enum.DECISION_TREE_REGRESSOR:
            return DecisionTreeRegressor(*args, **kwargs)
        elif self == MachineLearning_Enum.RANDOM_FOREST_REGRESSOR:
            return RandomForestRegressor(*args, **kwargs)
        elif self == MachineLearning_Enum.GRADIENT_BOOSTING_REGRESSOR:
            return GradientBoostingRegressor(*args, **kwargs)

        elif self == MachineLearning_Enum.KMEANS:
            return KMeans(*args, **kwargs)
        elif self == MachineLearning_Enum.DBSCAN:
            return DBSCAN(*args, **kwargs)
        else:
            raise ValueError(f"지원하지 않는 알고리즘입니다: {self}")
    
    def fit(self, X:list, y:list, model:any):
        model.fit(X, y)
        
classifier_models = {
                MachineLearning_Enum.LOGISTIC_REGRESSION,
                MachineLearning_Enum.KNN_CLASSIFIER,
                MachineLearning_Enum.SVM_CLASSIFIER,
                MachineLearning_Enum.DECISION_TREE_CLASSIFIER,
                MachineLearning_Enum.RANDOM_FOREST_CLASSIFIER,
                MachineLearning_Enum.GRADIENT_BOOSTING_CLASSIFIER,
                MachineLearning_Enum.NAIVE_BAYES
            }
regression_models = {
            MachineLearning_Enum.LINEAR_REGRESSION,
            MachineLearning_Enum.KNN_REGRESSOR,
            MachineLearning_Enum.SVR,
            MachineLearning_Enum.DECISION_TREE_REGRESSOR,
            MachineLearning_Enum.RANDOM_FOREST_REGRESSOR,
            MachineLearning_Enum.GRADIENT_BOOSTING_REGRESSOR
        }
clustering_models ={
            MachineLearning_Enum.KMEANS,
            MachineLearning_Enum.DBSCAN
        }

def helper_name_to_model_enum(model_name: str) -> MachineLearning_Enum:
    """문자열 모델 이름을 MachineLearning_Enum으로 변환"""
    for enum_member in MachineLearning_Enum:
        if enum_member.value.lower() == model_name.lower():
            return enum_member
    raise ValueError(f"'{model_name}'에 해당하는 MachineLearning_Enum이 없습니다.")



class ML_y_types(Enum):
    raw = 0 # 그대로 씀 ( y값이 정수 또는 실수 형식여야함 )
    label = 1 # 라벨 사용요구. (이떄 모델은 분류기여야한다) y값이 분류형이며, 문자열이 들어간 경우 라벨 인코딩을 요구하는 것으로 해석

def helper_name_to_ML_y_types(t_type_name:str)->ML_y_types:
    if t_type_name == "label":
        return ML_y_types.label
    elif t_type_name == "raw":
        return ML_y_types.raw
    else:
        raise f"idk y type -> {t_type_name}"

def _make_dir(dir:str):
        os.makedirs(dir, exist_ok=True)
        
def _save_object(obj: any, filefullpath: str):
    #filepath = os.path.join(saved_dir, filename)
    with open(filefullpath, "wb") as f:
        # [수정] pickle.dump()를 사용하여 파일에 직접 저장
        pickle.dump(obj, f, protocol=5)
    print(f"  - Saved: {filefullpath}")

def load_file(filepath:str)->any:
    output = None
    with open(filepath, 'rb') as f:
        output = pickle.load(f)
    return output

# 클러스터링용
def Make_Clustering_sets(id: str, StoragePath: str, Original_X: list):
    """
    클러스터링용 데이터셋 준비
    - 전체 X를 표준화(StandardScaler)만 수행
    - y는 필요 없음
    """
    saved_dir = os.path.join(StoragePath, id)
    _make_dir(saved_dir)

    # X 표준화
    X_Standarder = StandardScaler()
    X_scaled = X_Standarder.fit_transform(Original_X)
    _save_object(X_Standarder, os.path.join(saved_dir, "x_standarder.pkl"))

    return X_scaled


# 회귀 / 분류 용
def Make_Train_Test_sets(
    id: str, StoragePath: str, 
    model_enum:MachineLearning_Enum, Original_X:list, Original_y:list, y_type:ML_y_types, 
    
    *args, **kwargs # Args for the train_test_split()
):
    
    # Make Save Path ( abs )
    saved_dir = os.path.join(StoragePath, id)
    # make directory
    _make_dir(saved_dir)
    
    
    # [1/3] - y Data check
    checked_y:list = []
    if(y_type == ML_y_types.label):
            # 분류기 모델로 선택했는 지 검증하고, LabelEncoder 사용
            
            if model_enum not in classifier_models:
                raise "분류기가 아니므로 실패"
            
            Labeler = LabelEncoder()
            y_encoded = Labeler.fit_transform(Original_y)
            
            # 라벨러 저장
            _save_object(Labeler, os.path.join(saved_dir, "y_label_encoder.pkl"))
            checked_y = y_encoded
    else:
        # 예측/군집화등인지 체크 ( 분류기 모델 적발시 실패 )
        
        if model_enum in classifier_models:
            # 분류기 모델 적발 된
            raise "회귀 및 군집형은 분류기 금지"
        checked_y = Original_y
        
    # [2/3] - Train/Test Data create
    X_train, X_test, y_train, y_test = train_test_split(Original_X, checked_y, *args, **kwargs)
    
    
    # [3/3] Standard-Encoding X data(Xtrain + Xtest)
    checked_x:list = []
    
    X_Standarder = StandardScaler()
    standarded_X = X_Standarder.fit_transform( X_train )
    
    _save_object(X_Standarder, os.path.join(saved_dir, "x_standarder.pkl"))
    
    standarded_X_test =  X_Standarder.transform(X_test) # 테스트 데이터도 알맞게 정규화
    
    
    return standarded_X, standarded_X_test, y_train, y_test
            
        
def Model_fit(
    id: str, StoragePath: str, model_enum: MachineLearning_Enum, 
    X_train, X_test=None, y_train=None, y_test=None, 
    *args, **kwargs
) -> Optional[float]:
    """
    모델 종류에 따라 자동으로 학습 및 평가 수행
    - 분류기: accuracy_score
    - 회귀기: R² score
    - 군집화: silhouette_score (y_test는 무시)
    """
    try:
        saved_dir = os.path.join(StoragePath, id)
        
        
        print("?")
        model = model_enum.create_model(*args, **kwargs)
        
       
        print(f"[Model_fit] 모델 생성 완료: {model_enum.value}")
        

        # ===== 분류(Classification) =====
        if model_enum in classifier_models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            print(f"[Classification] 정확도: {acc:.4f}")
            print(classification_report(y_test, y_pred))
            
            _save_object(model, os.path.join(saved_dir, "ML_Model.pkl"))
            return acc

        # ===== 회귀(Regression) =====
        elif model_enum in regression_models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            print(f"[Regression] R²: {r2:.4f}, MSE: {mse:.4f}")
            
            _save_object(model, os.path.join(saved_dir, "ML_Model.pkl"))
            
            result_array = np.nan_to_num(r2, nan=0.0, posinf=1e10, neginf=-1e10) # Nan -> 0.0 처리
            
            print(f"r2 -> {result_array}")
            return result_array

        # ===== 군집화(Clustering) =====
        elif model_enum in clustering_models:
            model.fit(X_train)
            labels = model.labels_

            # DBSCAN의 경우 noise(-1)가 많을 때 silhouette 계산 불가능할 수 있음
            unique_labels = set(labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:
                score = silhouette_score(X_train, labels)
                print(f"[Clustering] 실루엣 점수: {score:.4f}")
                
                _save_object(model, os.path.join(saved_dir, "ML_Model.pkl"))
                #print(f"SCORE -> {score}")
                return score
            else:
                print("[Clustering] 유효한 클러스터 수 부족 (silhouette 계산 불가)")
                return None

        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_enum}")

    except Exception as e:
        print(f"[Model_fit] 에러 발생: {e}")
        raise e
        
def ML_Predict(id: str, StoragePath: str, Original_X):
    saved_dir = os.path.join(StoragePath, id)
    
    # Model Load
    Model = load_file( os.path.join(saved_dir, "ML_Model.pkl") )
    
    # X Standarder Load
    X_Standarder:StandardScaler = load_file( os.path.join(saved_dir, "x_standarder.pkl") )
    X_test = X_Standarder.transform(Original_X)
    
    
    # [opt] y Labeler Load
    labeler_path = os.path.join(saved_dir, "y_label_encoder.pkl")
    has_labeler = os.path.exists(labeler_path)

    # ========== [1] 분류기 (Classifier) ==========
    if has_labeler:
        y_Labeler = load_file(labeler_path)
        classes: np.array = y_Labeler.classes_

        if hasattr(Model, "predict_proba"):
            y_pred_proba = Model.predict_proba(X_test)[0]

            Output = {
                "argmax": {},
                "argmin": {},
                "all": {}
            }

            max_index = np.argmax(y_pred_proba)
            min_index = np.argmin(y_pred_proba)

            for i, class_label in enumerate(classes):
                if i == max_index:
                    Output["argmax"] = {str(class_label): float(y_pred_proba[i])}
                if i == min_index:
                    Output["argmin"] = {str(class_label): float(y_pred_proba[i])}
                Output["all"][str(class_label)] = float(y_pred_proba[i])

            return Output

        else:
            # 일부 분류기 (ex. SVM) 은 predict_proba 없음 → predict로만 처리
            y_pred = Model.predict(X_test)
            decoded = y_Labeler.inverse_transform(y_pred)
            return {"predicted_label": decoded.tolist()}

    # ========== [2] 회귀기 (Regressor) ==========
    elif hasattr(Model, "predict") and not hasattr(Model, "labels_"):
        y_pred = Model.predict(X_test)
        Output = {
            "predicted_values": y_pred.tolist(),
            "mean": float(np.mean(y_pred)),
            "std": float(np.std(y_pred))
        }
        return Output

    # ========== [3] 군집화 (Clustering) ==========
    elif hasattr(Model, "labels_") or hasattr(Model, "fit_predict"):
        # predict가 없을 수도 있음(DBSCAN 등)
        try:
            y_pred = Model.predict(X_test)
        except Exception:
            y_pred = Model.fit_predict(X_test)

        unique_labels = np.unique(y_pred)
        cluster_counts = {int(lbl): int(np.sum(y_pred == lbl)) for lbl in unique_labels}

        Output = {
            "clusters": y_pred.tolist(),
            "unique_labels": unique_labels.tolist(),
            "cluster_counts": cluster_counts
        }
        return Output

    # ========== [4] 알 수 없는 모델 ==========
    else:
        raise ValueError("알 수 없는 모델 유형입니다. 학습된 모델 파일을 확인하세요.")