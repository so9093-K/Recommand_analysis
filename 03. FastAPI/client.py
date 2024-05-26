import numpy as np
import pandas as pd
import requests
from pandas import json_normalize
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
from tqdm.notebook import tqdm
from sklearn.utils import shuffle


# API로부터 받아온 데이터 셋을 변환, 정제하여 주는 클래스 
class Load_Dataset :

    # 클래스 Load_Dataset을 지정할 시 실행되는 함수
    def __init__(self) :                         
        print("데이터 서버 연결중 ....")
        payload = {'key1':'value1','key2':'value2'}
        headers = {'Content-Type':'application/json; charset=utf-8'}
        read_request = requests.get('http://192.168.1.55:8000/api/train/user-novel-grade',params=payload, headers=headers)
        read_request = read_request
        rjson = read_request.json()    
        df = json_normalize(rjson['data'])   # 받아온 API 속의 Json 데이터를 판다스의 데이터프레임으로 변환
        self.df = df                         # 데이터 프레임 참조 매개변수 지정 
        print("데이터 서버 연결.... ")

    # CF 모델 결과를 만들어내기 위한 데이터 만들기
    def cf_export_df(self) :  
        return self.df

    # NCF모델 학습하기 위한 데이터 만들기
    def ncf_export_df(self) :  
        df = self.df.dropna()
        df = df.loc[df.grade != 0] 

        unique_user_lst = list(np.unique(df['user_id'])) 
        sample_num = len(unique_user_lst)               
        sample_user_idx = np.random.choice(len(unique_user_lst),sample_num, replace = False) 
        sample_user_lst = [unique_user_lst[idx] for idx in sample_user_idx] 
        df = df[df['user_id'].isin(sample_user_lst)] 
        df = df.reset_index(drop=True)               

        df_count = df.groupby(['user_id']).count()   
        df['count'] = df.groupby('user_id')['user_id'].transform('count') 
        df = df[df['count'] >1 ]

        df_train,df_test = self.train_test_split(df)

        users = list(np.sort(df.user_id.unique()))
        novels = list(np.sort(df.novel_id.unique()))

        train_rows = df_train['user_id'].astype(int)
        train_cols = df_train['novel_id'].astype(int)
        train_values = list(df_train.grade)

        train_uids = np.array(train_rows.tolist())
        train_nids = np.array(train_cols.tolist())

        real_rows = df['user_id'].astype(int)
        real_cols = df['novel_id'].astype(int)
        real_values = list(df.grade)

        real_uids = np.array(real_rows.tolist())
        real_nids = np.array(real_cols.tolist())

        train_df_neg = self.get_negatives(train_uids,train_nids,novels,df_test)
        real_df_neg = self.get_negatives(real_uids,real_nids,novels,df)

        return users, novels, df_test, train_uids, train_nids, real_uids, real_nids, train_df_neg, real_df_neg

    # 데이터 프레임 속에 첫 번째 데이터를 0으로 만들기   데이터 전처리에 사용되는 함수 중의 하나
    def mask_first(self,x) : 
        result = np.ones_like(x)
        result[0] = 0
        return result

    # NCF 모델 학습에 사용되는 Train, Test 데이터 셋 생성 함수
    def train_test_split(self,df) :
        df_test = df.copy(deep=True)
        df_train = df.copy(deep=True)

        df_test = df_test.groupby(['user_id']).first()
        df_test['user_id'] = df_test.index
        df_test = df_test[['user_id','novel_id','grade']]
        df_test = df_test.reset_index(drop=True)

        mask = df.groupby(['user_id'])['user_id'].transform(self.mask_first).astype(bool)
        df_train = df.loc[mask]
        return df_train, df_test

    # NegativeList 데이터 생성 (유저가 구매한 제품, 구매하지 않은 제품을 나열한 데이터 프레임,  1= 구매한제품, 0=구매하지 않은 제품)
    # ex (userID,구매한제품), 구매하지 않은 제품, 구매하지 않은 제품, 구매하지 않은 제품...............
    def get_negatives(self,uids,nids,novels,DataFrame) :
        negativeList = []
        list_u = DataFrame['user_id'].values.tolist()
        list_n = DataFrame['novel_id'].values.tolist()

        df_ratings = list(zip(list_u,list_n))
        zipped = set(zip(uids,nids))
        for (u,n) in tqdm(df_ratings) :
            negatives = []
            negatives.append((u,n))
            for t in range(15) :
                j = np.random.randint(len(novels))
                while (u,j) in zipped :
                    j = np.random.randint(len(novels))
                negatives.append(j)
            negativeList.append(negatives)
        df_neg = pd.DataFrame(negativeList)
        return df_neg

    # Numpy로 변환된 학습 데이터 생성   - 유저,소설,num_neg, 소설개수를 인자로 받음
    def get_train_instances(self, uids, nids, num_neg, num_novels) :

        user_input, novel_input, labels = [], [], []
        zipped = set(zip(uids,nids))
        for (u,n) in zip(uids,nids) :
            user_input.append(u)
            novel_input.append(n)
            labels.append(1)
            for t in range(num_neg) :
                j = np.random.randint(num_novels)
                while (u,j) in zipped :
                    j = np.random.randint(num_novels)

                user_input.append(u)
                novel_input.append(j)
                labels.append(0)
        return user_input, novel_input, labels

# 제품간의 유사성을 기반으로 추천해주는 기능을 만들어주는 클래스 지정
class Content_based_model :

    # Content_based_model 클래스가 지정 될때 실행되는 코드   self. 입력시 다른 함수에서 참조할 수 있다.
    def __init__(self) :  
        ratings = Load_Dataset().cf_export_df()
        novel_list = ratings['novel_id'].unique().tolist()
        self.novel_list = novel_list
        ratings_matrix = ratings.pivot_table('grade',index='user_id',columns = 'novel_id')
        ratings_matrix = ratings_matrix.fillna(0)
        ratings_matrix_T = ratings_matrix.T
        item_sim = cosine_similarity(ratings_matrix_T,ratings_matrix_T)
        item_sim_df = pd.DataFrame(data = item_sim, index = ratings_matrix.columns,
                            columns = ratings_matrix.columns)
        self.item_sim_df = item_sim_df

    # 모든 유저의 추천 리스트 생성 함수  (API)
    def create_recommend_data_similarity(self) :
        recommend_predict_novels_ =pd.DataFrame()

        for i in tqdm(range(len(self.novel_list))):
            result = pd.Series.to_frame(self.item_sim_df[self.novel_list[i]].sort_values(ascending=False)[:6])
            result.reset_index(inplace=True)
            origin_novel = result.novel_id[0]
            recommend_novel = result.novel_id[1:].values.tolist()
            prediction = result[result.novel_id[0]][1:].values.tolist()
            result = result[1:]
            result['origin_novel_id'] = origin_novel
            result['recommend_novel_id'] = recommend_novel
            result['predictions'] = prediction
            result['origin_novel_id'] = result['origin_novel_id'].astype(int)
            result['recommend_novel_id'] = result['recommend_novel_id'].astype(int)
            dict_ = result[['origin_novel_id','recommend_novel_id','predictions']]
            recommend_predict_novels_ = pd.concat([recommend_predict_novels_,dict_])
        recommend_predict_novels_.reset_index(inplace = True, drop = True)
        dict_ = recommend_predict_novels_.T.to_dict()
        for i in range(len(dict_)) :
            dict_[i]['origin_novel_id'] = int(dict_[i]['origin_novel_id'])
            dict_[i]['recommend_novel_id'] = int(dict_[i]['recommend_novel_id'])
        dict_ = solution(dict_)
        return dict_

# 데이터 전처리에 사용되는 클래스 (API로 보내주는 데이터를 만들어낼 때 사용)
class person(object):
    def __init__(self,name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return "'"+self.name+"'"
# 데이터 전처리에 사용되는 함수 (API로 보내주는 데이터를 만들어낼 때 사용)
def solution(dict_):
    tmp_dict = {}
    dict_key_list = list(dict_.keys())
    for i in range(len(dict_key_list)) :
        tmp_dict[person("trains[]")] = str(dict_[dict_key_list[i]]).replace("""'""",'''"''')

#     print (tmp_dict)
    return tmp_dict


# 텐서플로우 
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model   


# Neural Collaborative Filtering 알고리즘을 만들고 학습하고 결과를 뽑아주는 클래스 지정 
class Create_Base_NeuMF :

    # Create_Base_NeuMF 클래스가 지정될 때 실행되는 코드들 (유저,소설 관련 데이터, NegativeList 등)
    def __init__(self) :
        self.users, self.novels, self.df_test, self.train_uids, self.train_nids, self.real_uids, self.real_nids,\
        self.train_df_neg, self.real_df_neg = Load_Dataset().ncf_export_df()

    # 모델 생성 함수 
    def create_model(self,users,novels) :
        latent_features = 8
        # Input
        user = Input(shape=(1,), dtype='int32')
        novel = Input(shape=(1,), dtype='int32')
        # User embedding for GMF
        gmf_user_embedding = Embedding(max(users)+1, latent_features, input_length=user.shape[1])(user)
        gmf_user_embedding = Flatten()(gmf_user_embedding)
        # Item embedding for GMF
        gmf_novel_embedding = Embedding(max(novels)+1, latent_features, input_length=novel.shape[1])(novel)
        gmf_novel_embedding = Flatten()(gmf_novel_embedding)
        # User embedding for MLP
        mlp_user_embedding = Embedding(max(users)+1, 32, input_length=user.shape[1])(user)
        mlp_user_embedding = Flatten()(mlp_user_embedding)
        # Item embedding for MLP
        mlp_novel_embedding = Embedding(max(novels)+1, 32, input_length=novel.shape[1])(novel)
        mlp_novel_embedding = Flatten()(mlp_novel_embedding)
        # GMF layers
        gmf_mul =  Multiply()([gmf_user_embedding, gmf_novel_embedding])
        # MLP layers
        mlp_concat = Concatenate()([mlp_user_embedding, mlp_novel_embedding])
        mlp_dropout = Dropout(0.2)(mlp_concat)        
        # Layer1    
        mlp_layer_1 = Dense(units=64, activation='relu', name='mlp_layer1')(mlp_dropout)  # (64,1)
        mlp_dropout1 = Dropout(rate=0.2, name='dropout1')(mlp_layer_1)                    # (64,1)
        mlp_batch_norm1 = BatchNormalization(name='batch_norm1')(mlp_dropout1)            # (64,1)
        # Layer2
        mlp_layer_2 = Dense(units=32, activation='relu', name='mlp_layer2')(mlp_batch_norm1)  # (32,1)
        mlp_dropout2 = Dropout(rate=0.2, name='dropout2')(mlp_layer_2)                        # (32,1)
        mlp_batch_norm2 = BatchNormalization(name='batch_norm2')(mlp_dropout2)                # (32,1)
        # Layer3
        mlp_layer_3 = Dense(units=16, activation='relu', name='mlp_layer3')(mlp_batch_norm2)  # (16,1)
        # Layer4
        mlp_layer_4 = Dense(units=8, activation='relu', name='mlp_layer4')(mlp_layer_3)       # (8,1)
        # merge GMF + MLP
        merged_vector = tf.keras.layers.concatenate([gmf_mul, mlp_layer_4])
        # Output layer
        output_layer = Dense(1, kernel_initializer='lecun_uniform', name='output_layer')(merged_vector) # 1,1 / h(8,1)초기화
        # Model
        self.model = Model([user, novel], output_layer)
        self.model.compile(optimizer= 'adam', loss= 'binary_crossentropy')
        self.model.build(user.shape)
        self.model.summary()

        return self.model

    # 모델 학습 함수 모든 수치형 데이터들을 Numpy Array형식으로 변환, 학습 후 모델 저장 (API)
    def train_model(self) :   

        model = self.create_model(self.users,self.novels)

        user_input, novel_input, labels_input = Load_Dataset().get_train_instances(self.train_uids,
                                    self.train_nids,3,len(self.novels))
        user_data_shuff, novel_data_shuff, label_data_shuff = \
        shuffle(user_input,novel_input,labels_input)
        user_data_shuff = np.array(user_data_shuff).reshape(-1,1)
        novel_data_shuff = np.array(novel_data_shuff).reshape(-1,1)
        label_data_shuff = np.array(label_data_shuff).reshape(-1,1)    
        model.fit([user_data_shuff, novel_data_shuff], label_data_shuff, epochs = 10, batch_size = 64, verbose = 1)                                  
        model.save('./data/NCF_model.h5')
        return print("모델 저장...")

    # 모델 불러오기 함수
    def load_model(self) :
        model = tf.keras.models.load_model('./data/NCF_model.h5')
        print("모델 로드...")
        return model
    
    # 모델을 사용하여 한 유저의 추천 목록을 생성해주는 함수
    def recommend_user_item(self, user_id, model) :
        # model = self.load_model()
        real_df = self.real_df_neg.copy()
        real_df.insert(0, 'users', pd.DataFrame(list(real_df[0]))[0])
        real_df.insert(1, 'novels', pd.DataFrame(list(real_df[0]))[1])        
        del real_df[0]
        ids = real_df[real_df['users']==user_id]
        list_ = ids[ids.columns[2:]].values
        list_ = list(np.array(ids[ids.columns[2:]].values).flatten().tolist())

        user_candidate_novel = np.array(list_).reshape(-1,1)
        user_input = np.full(len(user_candidate_novel), user_id, dtype='int32').reshape(-1,1)
        predictions = model.predict([user_input, user_candidate_novel])
        predictions = predictions.flatten().tolist()
        novel_to_pre_score = {novel[0]:pre for novel,pre in zip(user_candidate_novel,predictions)}
        novel_to_pre_score = dict(sorted(novel_to_pre_score.items(),key = lambda x:x[1], reverse = True))
        recommend_novel_lst = list(novel_to_pre_score.keys())
        return novel_to_pre_score , recommend_novel_lst

    # 모델을 사용하여 모든 유저의 모든 추천 목록을 생성해주는 함수 (API)
    def export_recommend_users_prediction(self, model) :
        export_df = pd.DataFrame()
        for i in range(len(self.users)) :
            novel_to_pre_score , recommend_novel_lst = self.recommend_user_item(self.users[i],model)
            result_df = pd.DataFrame()
            result_df['recommend_novel'] = list(novel_to_pre_score.keys())
            result_df['predictions_score'] = list(novel_to_pre_score.values())
            result_df.insert(0,'user_id',self.users[i])
            export_df = pd.concat([export_df,result_df])
        return export_df



# 모델 평가 (API 따로 없음)
import heapq

class Metric:

    def __init__(self):
        pass
    
    # 모델을 사용하여 추천 목록을 생성하였을 때, 기존의 Test 데이터를 활용하여 데이터 데이터에 있는 유저가 구매한 소설이  모델을 사용하여 뽑아낸 추천목록 속에 있을 때 1로 지정
    def get_hits(self, k_ranked, holdout):
        """
        hit 생성 함수
        hit := holdout(df_test의 item)이 K순위 내에 있는지 여부
        """
        for item in k_ranked:
            if item == holdout:
                return 1
        return 0
    
    def eval_rating(self, idx, ratings, negetives, K, model) :
        """
        모든 유저의 예측 목록에서 각 유저마다 positive novel이 K 순위 내에 얼마나 많이 있는지 평가하는 함수
        """
        novels = negetives[idx]          # negative items [neg_item_id, ... ] (1,100)
        user_idx = ratings[idx][0]      # [user_id, item_id][0]
        holdout = ratings[idx][1]       # [user_id, item_id][1]
        novels.append(holdout)           # holdout 추가 [neg_item_id, ..., holdout] (1,101)
        
        # prediction
        
        predict_user = np.full(len(novels), user_idx, dtype='int32').reshape(-1,1) # [[user_id], ...], (101, 1)
        np_novels = np.array(novels).reshape(-1,1)                                  # [[item_id], ... ], (101, 1)
        
        # 모델 사용 
        predictions = model.predict([predict_user, np_novels])
        predictions = predictions.flatten().tolist()
        item_to_pre_score = {item:pre for item,pre in zip(novels, predictions)}
        
        # 점수가 높은 상위 K개 아이템 리스트 생성
        k_ranked = heapq.nlargest(K, item_to_pre_score, key = item_to_pre_score.get)
        
        # holdout이 상위 K 순위에 포함 되는지 체크
        # { 1 : 포함 , 0 : 안포함}
        hits = self.get_hits(k_ranked, holdout)
        
        return hits   

    def evaluate_top_k(self,df_neg, df_test, model, K=5) :
        """
        TOP-K metric을 사용해 모델을 평가하는 함수
        """
        hits = []
        test_u = df_test['user_id'].values.tolist()
        test_n = df_test['novel_id'].values.tolist()
        
        test_ratings = list(zip(test_u, test_n))
        df_neg = df_neg.drop(df_neg.columns[0], axis = 1)
        test_negetives = df_neg.values.tolist() #[[(user_id, item_id =holdout)], neg_item,neg_item,....]
        
        # user 샘플링
    #     sample_idx_lst = np.random.choice(len(test_ratings), int(len(test_ratings) * 0.3))
        for user_idx in tqdm(range(len(test_ratings))) :
            hitrate = self.eval_rating(user_idx, test_ratings, test_negetives, K, model)
            hits.append(hitrate) # ex. [1,0,1,1,0,....] (1, df_test.shape[0])
        
        return hits

    # 총 점수 생성   -    모든 유저의 추천목록을 대상으로 평가
    def calculate_top_k_metric(self,df_neg,df_test,model) :
        hit_lst = self.evaluate_top_k(df_neg,df_test,model,K=5)
        top_k_metric = np.mean(hit_lst)
        print("metric : {}".format(top_k_metric))
        return top_k_metric


