import os 

import pandas as pd
import random
import numpy as np
dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}



def custom_train_test_split(df, ratio=0.9, split=True):
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.seed(42)
    random.shuffle(users)
    
    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test
    
def custom_train_test_split_2(df, ratio=0.9, split=True):
    df_test = df[(df.answerCode == -1)]
    random.seed(42)
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    users = (list(set(df['userID'].value_counts().index) - set(df_test.userID)))
    random.shuffle(users)
    
    user_ids = random.sample(users,744)
    
    train = df[df['userID'].isin(user_ids)== False]
    test = df[df['userID'].isin(user_ids)]
    
    _test = test[test['userID'] == test['userID'].shift(-1)]
    train = pd.concat([train,_test])
    test = test[test['userID'] != test['userID'].shift(-1)]
    
    return train, test


def make_dataset(train, valid):

    # X, y 값 분리
    y_train = train["answerCode"]
    x_train = train.drop(["answerCode"], axis=1)

    y_valid = valid["answerCode"]
    x_valid = valid.drop(["answerCode"], axis=1)

    return y_train, x_train, y_valid, x_valid