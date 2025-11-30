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


WithId_STATUS = \
"""
{
    "id": "EDR-AI",
    "print_type": "detail"
}
"""
print( requests.post( "http://"+API_SERVER_IP+":"+str(API_SERVER_PORT)+WithId_Status_PATH, data=json.dumps(WithId_STATUS)).content )
