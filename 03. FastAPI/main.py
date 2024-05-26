import client
from starlette.requests import Request
from fastapi import FastAPI
import subprocess
import requests
import time
import atexit
from fastapi_utils.tasks import repeat_every
import schedule

app = FastAPI()

@app.get('/')
def index() :
    return "API.."

@app.post('/api/train/save_ncf_model')  # 요청이 오면 학습하고 저장
def train_model() :
     Class_NeuMF.train_model()
     return "모델 학습..저장.."

@app.post('/api/export/result_ContentMF') # 제품별 추천 리스트 보내주기
def export_content() : 
    result = Class_ContentMF.create_recommend_data_similarity()
    print(result)
    return dict({'result':'suceess','data': list(result.values())})

@app.post('/api/export/result_NeuMF')  # 유저별 추천 리스트 보내주기
def export_ncf() :
    result = Class_NeuMF.export_recommend_users_prediction(model)
    result.reset_index(inplace = True, drop = True)
    result = result.T.to_dict()
    print(result)
    # return "good"
    return dict({'result':'suceess','data': list(result.values())})

if __name__ == 'main' :

    Class_NeuMF = client.Create_Base_NeuMF()
    model = Class_NeuMF.load_model()
    Class_ContentMF = client.Content_based_model()
    # data = client.Load_Dataset()

