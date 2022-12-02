import os
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from feature_cfg import FE

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        self.cat_features = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_features(self):
        return self.cat_features

    def __preprocessing(self, df, cat_features):
        df = df.drop(FE.drop, axis = 1)
        df = df[~(df.answerCode == -1)]
        tmp = df.columns.to_list()

        if self.args.model == 'catboost':
            # 3이상 1000이하 개수의 범주형 데이터 고르기
            for i in tmp:
                # print(f'{i} = {df[i].nunique()}')
                num = int(df[i].nunique())
                if 3< num < 1000:
                    cat_features.append(i)
            # feature 추가
            cat_features += [ 'assessmentItemID',
                        'testId',]

            # 라벨 인코딩
            le = LabelEncoder()

            for cat in cat_features:
                if df[cat].dtypes != "int": 
                    df[cat] = le.fit_transform(df[cat])
                df[cat] = df[cat].astype("category")

        if self.args.model == 'lgbm':

            le = LabelEncoder()

            lgbm_col = ['testId', 'assessmentItemID']
            for lgbm in lgbm_col:
                df[lgbm] = le.fit_transform(df[lgbm])

        return df, cat_features

    def __feature_engineering(self, df):

        
        """
        feature 정보

        "user_mean", 'user_sum' : 유저별 정답률, 푼 문제 수
        "tag_mean", 'tag_sum' : 태크별 정답률, 맞은 문제 수
        "Item_mean", 'Item_sum' : 문제별 정답률, 맞은 문제 수
        "test_mean", 'test_sum' : 시험지별 정답률, 맞은 문제 수
        'before_ans_rate_testid_byuser' : 유저별 이전 시험지 정답률
        'user_test_mean' : 유저별 시험지 정답률 (이거 사용하면 안됨)
        'assessmentItemID_shift' : 유저별 이전 문제 맞은 수 누적 
        'Item_mean_cum' : 유저별 이전 문제 정답률 누적
        'problem_time' : 유저별 이전 문제를 푸는데 걸린 시간
            'year', 'month', 'day', 'hour', 'minute', 'second' : 문제 푼 연월일 시간 
            'MA_prec_*' : 이동평균 정답률
        """
        
        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)
        
        ## 이전 시험지 정답률
        df['answerCode_copy'] = df['answerCode'].copy()
        df.loc[df['answerCode'] == -1,'answerCode'] = 0
        df['user_test_mean'] = df.loc[:, ['userID','testId','answerCode']].groupby(['userID','testId']).transform(np.mean)
        sample_df = df.loc[:, ['userID','testId','answerCode']].groupby(['userID','testId']).mean().groupby(['userID']).shift(1).reset_index(inplace=False)
        sample_df.columns = ['userID','testId','before_ans_rate_testid_byuser']
        df = df.merge(sample_df,how='inner',on=['userID','testId'])
        df.loc[df['before_ans_rate_testid_byuser'].isna(),'before_ans_rate_testid_byuser'] = df.loc[df['before_ans_rate_testid_byuser'].isna(),'user_test_mean']
        del df['user_test_mean']
        # 시험지별 정답률
        correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
        correct_t.columns = ["test_mean", 'test_sum']
        # 유저별 정답률
        correct_u = df.groupby(['userID'])['answerCode'].agg(['mean', 'count'])
        correct_u.columns = ["user_mean", 'user_sum']
        # tag 별 정답률
        correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
        correct_k.columns = ["tag_mean", 'tag_sum']
        # 문제 별 정답률 
        correct_i = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum'])
        correct_i.columns = ["Item_mean", 'Item_sum']
        
        df['down'] = df.loc[:, ['userID','testId','answerCode']].groupby(['userID','testId']).shift(1).fillna(1)

        df = pd.merge(df, correct_u, on=['userID'], how="left")
        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        df = pd.merge(df, correct_i, on=['assessmentItemID'], how="left")
        
        
        df['cnt'] = 1
        df['cnt'] = df.loc[:, ['userID','testId','cnt']].groupby(['userID','testId']).shift(1).fillna(0)
        df['cnt'] = df.loc[:, ['userID','testId','cnt']].groupby(['userID','testId']).cumsum()
        
        
        ## 이전 문제 정답률(누적)
        df['assessmentItemID_shift'] = df.loc[:, ['userID','testId','answerCode']].groupby(['userID','testId']).shift(1).fillna(0)
        df['assessmentItemID_shift'] = df.loc[:, ['userID','testId','assessmentItemID_shift']].groupby(['userID','testId']).cumsum()
        df['Item_mean_cum'] = (df['assessmentItemID_shift']/df['cnt']).fillna(0.747278741536055)
        del df['cnt']

        # 유저가 이전 문제를 푸는데 걸린 시간
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        ## 병웅 피쳐
        df['weekday'] = df['Timestamp'].dt.weekday
        df['assessment_main'] = df['assessmentItemID'].apply(lambda x : int(x[2]))
        df['assessment_mid'] = df['assessmentItemID'].apply(lambda x : int(x[4:7]))
        df['assessment_sub'] = df['assessmentItemID'].apply(lambda x : int(x[-2:]))

        diff = df.loc[:, ['userID','testId', 'Timestamp']].groupby(['userID','testId']).diff(1).fillna(pd.Timedelta(seconds=0))
        df['problem_time'] = diff['Timestamp']
        df['problem_time'] = df['problem_time'].dt.seconds

            
        df.loc[df['problem_time'] > 3600,'problem_time'] = 3600

        ## assessment_main 분류 정답률, 문제풀이 시간과 병합(count, mean기준으로 feature 추가)
        assessment_main_features = df.groupby(['userID','assessment_main']).agg({
        'answerCode':['mean', 'count'],
        'problem_time':['mean', 'count']
        })
        assessment_main_features.reset_index(inplace=True)
        assessment_main_features.columns = ["userID","assessment_main","assessment_main_mean","assessment_main_count", "assessment_main_problem_mean", "assessment_main_problem_count"]
        df = pd.merge(df, assessment_main_features, on=['userID', 'assessment_main'], how = 'left')


        ## 'assessment_mid' assess, 분류 정답률, 문제풀이 시간과 병합(count, mean기준으로 feature 추가)
        assessment_mid_features = df.groupby(['userID','assessment_mid']).agg({
        'answerCode':['mean', 'count'],
        'problem_time':['mean', 'count']
        })
        assessment_mid_features.reset_index(inplace=True)
        assessment_mid_features.columns = ["userID","assessment_mid","assessment_mid_mean","assessment_mid_count", "assessment_mid_problem_mean", "assessment_mid_problem_count"]
        df = pd.merge(df, assessment_mid_features, on=['userID', 'assessment_mid'], how = 'left')

        ## 'assessment_sub' assess, 분류 정답률, 문제풀이 시간과 병합(count, mean기준으로 feature 추가)
        assessment_sub_features = df.groupby(['userID','assessment_sub']).agg({
        'answerCode':['mean', 'count'],
        'problem_time':['mean', 'count']
        })
        assessment_sub_features.reset_index(inplace=True)
        assessment_sub_features.columns = ["userID","assessment_sub","assessment_sub_mean","assessment_sub_count", "assessment_sub_problem_mean", "assessment_sub_problem_count"]
        df = pd.merge(df, assessment_sub_features, on=['userID', 'assessment_sub'], how = 'left')


            # 문제를 푸는 시간대
        time_grid = ['year', 'month', 'day', 'hour', 'minute', 'second'] # 요일은 weekday 0~6
        for tg in time_grid:
            df[tg] = df['Timestamp'].transform(lambda x: getattr(pd.to_datetime(x), tg))
        # 요일 - 위에서 같이 돌리니 아웃풋이 이상하게 나와서 아래로 뺌
        df['weekday'] = pd.to_datetime(df['Timestamp']).dt.weekday
    
        # 사용자의 주 활동 시간
        mode_dict = df.groupby(['userID'])['hour'].agg(lambda x: pd.Series.mode(x)[0]).to_dict()
        df['hour_mode'] = df['userID'].map(mode_dict)
    

        
            # problem_time ; 1시간 즉 3600초 지나거나, 0초이면 그냥 nan으로 바꾸기
        df.loc[(df['problem_time'] == 0) | (df['problem_time'] > 3600),'problem_time'] = np.nan


        ### Rolling Window features ###
        #### answer rate over past period #### 
        # 숫자보다 이전 데이터가 적은 경우, 그냥 적은 데이터에서 계산된 mean 값을 쓴다.
        # for t in [5, 10, 20, 40, 80, 160, 320]:
        #     df['MA_prec_all_over'+str(t)] = df.groupby('userID')['answerCode'].transform(lambda x: x.rolling(t, 1, closed='left').mean())
        # # 시험지 분류에 따른 이동평균
        # for t in [5, 10, 50, 100]: 
        #     df['MA_prec_by_testmain'+str(t)] = df.groupby(['userID', 'test_main'])['answerCode'].transform(lambda x: x.rolling(t, 1, closed='left').mean())
        # for t in [3, 6, 10]: 
        #     df['MA_prec_by_testsub'+str(t)] = df.groupby(['userID', 'test_sub'])['answerCode'].transform(lambda x: x.rolling(t, 1, closed='left').mean())
        # # 문항 분류에 따른 이동평균
        # for t in [5, 10, 50, 100]: 
        #     df['MA_prec_by_assessmain'+str(t)] = df.groupby(['userID', 'assessment_main'])['answerCode'].transform(lambda x: x.rolling(t, 1, closed='left').mean())
        # for t in [3, 6]: 
        #     df['MA_prec_by_assesssub'+str(t)] = df.groupby(['userID', 'assessment_sub'])['answerCode'].transform(lambda x: x.rolling(t, 1, closed='left').mean())
        # # 시험지 번호에 따른 이동평균
        # for t in [3, 6]:
        #     df['MA_prec_by_testid'+str(t)] = df.groupby(['userID', 'testId'])['answerCode'].transform(lambda x: x.rolling(t, 1, closed='left').mean())
            
        df['answerCode'] = df['answerCode_copy']
		
        return df

    def load_data_from_file(self, file_name):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])
        # df = self.__feature_engineering(df)
        cat_features = []
        df, cat_features = self.__preprocessing(df, cat_features) ##config 파일로 feature 처리
        return df, cat_features

    def load_train_data(self, file_name):
        self.train_data, self.cat_features = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name)

    