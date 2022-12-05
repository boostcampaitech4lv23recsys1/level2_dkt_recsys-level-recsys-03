import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None
        csv_file_path = os.path.join(self.args.data_dir, "test_data.csv")
        self.df_test = pd.read_csv(csv_file_path) 

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.9, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = 744#int(len(data) * ratio)
        data_1 = data[:-size]
        data_2 = data[-size:]
        # random.seed(seed)
        # users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
        # a = df['userID'].value_counts().reset_index()
        # users = (list(set(df['userID'].value_counts().index) - set(self.df_test.userID)))
        # random.shuffle(users)
        
        # user_ids = random.sample(users,744)
        
        # train = df[df['userID'].isin(user_ids)== False]
        # test = df[df['userID'].isin(user_ids)]
        
        # _test = test[test['userID'] == test['userID'].shift(-1)]
        # train = pd.concat([train,_test])
        # test = test[test['userID'] != test['userID'].shift(-1)]
        

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        # def convert_time(s):
        #     timestamp = time.mktime(
        #         datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
        #     )
        #     return int(timestamp)

        # df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self,df):
        #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
        df.sort_values(by=['userID','Timestamp'], inplace=True)
        
        ## 이전 시험지 정답률
        df['answerCode_copy'] = df['answerCode'].copy()
        df.loc[df['answerCode'] == -1,'answerCode'] = 0
        sample_df = df[['userID','testId','answerCode']].groupby(['userID','testId']).mean().groupby(['userID']).shift(1).reset_index(inplace=False)
        sample_df.columns = ['userID','testId','test_answer_shift']
        df = df.merge(sample_df,how='inner',on=['userID','testId'])
        
        
        df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
        df['user_cum_prob'] = (df['user_correct_answer']/df['user_total_answer'])
        df.loc[df.user_cum_prob==np.inf,'user_cum_prob'] =np.nan
        
        df['down'] = df.loc[:, ['userID','testId','answerCode']].groupby(['userID','testId']).shift(1)


        ## 이전 문제 정답률(누적)
        df['user_Item_total_answer'] = df[['userID','testId','answerCode']].groupby(['userID','testId']).cumcount()
        df['user_Item_correct_answer'] = df[['userID','testId','answerCode']].groupby(['userID','testId']).shift(1)
        df['user_Item_correct_answer'] = df[['userID','testId','user_Item_correct_answer']].groupby(['userID','testId']).cumsum()
        df['user_Item_cum_prob'] = df['user_Item_correct_answer']/df['user_Item_total_answer'] 
        df.loc[df.user_Item_cum_prob==np.inf,'user_Item_cum_prob'] =np.nan
        
        # 유저가 이전 문제를 푸는데 걸린 시간
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        ## 병웅 피쳐
        df['assessment_main'] = df['assessmentItemID'].apply(lambda x : int(x[2]))
        df['assessment_mid'] = df['assessmentItemID'].apply(lambda x : int(x[4:7]))
        df['assessment_sub'] = df['assessmentItemID'].apply(lambda x : int(x[-2:]))

        diff = df[['userID','testId', 'Timestamp']].groupby(['userID','testId']).diff(1)
        df['problem_time'] = diff['Timestamp']
        df['problem_time'] = df['problem_time'].dt.seconds

        df.loc[(df['problem_time'] > 3600),'problem_time'] = 3600
        
        correct_t = df.groupby(['testId'])['answerCode'].agg(['mean','sum'])
        correct_t.columns = ["test_prob",'test_sum']

        correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean','sum'])
        correct_k.columns = ["tag_prob",'tag_sum']

        correct_i = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean','sum'])
        correct_i.columns = ["Item_prob",'Item_sum']

        df = pd.merge(df, correct_t, on=['testId'], how="left")
        df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
        df = pd.merge(df, correct_i, on=['assessmentItemID'], how="left")
    
            # problem_time ; 1시간 즉 3600초 지나거나, 0초이면 그냥 nan으로 바꾸기
        df.loc[(df['problem_time'] == 0) | (df['problem_time'] > 3600),'problem_time'] = np.nan
        df['answerCode'] = df['answerCode_copy']
        del df['answerCode_copy']
        del df['Timestamp']
        
        tag2idx = {j:i for i,j in enumerate(df['KnowledgeTag'].unique())}
        df['KnowledgeTag'] = df['KnowledgeTag'].map(tag2idx)
        Item2idx = {j:i for i,j in enumerate(df["assessmentItemID"].unique())}
        df["assessmentItemID"] = df["assessmentItemID"].map(Item2idx)  
        test2idx = {j:i for i,j in enumerate(df["testId"].unique())}
        df["testId"] = df["testId"].map(test2idx) 

        df = df[["userID",'KnowledgeTag',"assessmentItemID", "testId",
        'test_answer_shift', 'user_correct_answer', 'user_total_answer',
        'user_cum_prob', 'user_Item_total_answer',
        'user_Item_correct_answer', 'user_Item_cum_prob', 'problem_time',
        'test_prob', 'tag_prob', 'Item_prob', 'answerCode']]
        df = df.fillna(1)
        print(df)
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        df = self.__feature_engineering(df)
        # df = self.__preprocessing(df, is_train) ## FE에서 한번에 처리

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        self.args.n_questions = df["assessmentItemID"].nunique()
        self.args.n_test = df['testId'].nunique()
        self.args.n_tag = df['KnowledgeTag'].nunique()
        self.args.col_size = len(df.columns[4:-1])
        self.args.cate_size = len(df.columns[1:4])

        
        group = (df.groupby("userID").apply(lambda r: [r[i].values for i in df.columns[1:]]))


        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]
        # 각 data의 sequence length
        seq_len = len(row[0])

        # test, question, tag, correct = row[0], row[1], row[2], row[3]
        # KnowledgeTag, assessmentItemID, testId, user_mean, user_sum, Item_mean, Item_sum, test_mean, test_sum, tag_mean, tag_sum, problem_time, answerCode = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12]

        # cate_cols = [test, question, tag, correct]
        cate_cols = list(row)

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
    
        cate_cols.append(mask)
    
        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)
        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader
