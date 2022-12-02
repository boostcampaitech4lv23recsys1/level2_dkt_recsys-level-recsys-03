from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from feature_cfg import FE

import os





class CatBoost:

    def __init__(self, args):
        self.args = args
        self.model = CatBoostClassifier(
            iterations = 1500,
            random_seed = 42,
            learning_rate = 0.001,
            loss_function = 'Logloss', 
            eval_metric = 'AUC',
            early_stopping_rounds = 1000,
            use_best_model =  True,
            task_type = "GPU",
            bagging_temperature = 1,
            verbose = 50
            )


    def train(self, args, y_train, x_train, y_valid, x_valid, cat_features):
        train_pool = Pool(x_train[FE.features], y_train, cat_features = cat_features)
        eval_pool = Pool(x_valid[FE.features], y_valid, cat_features = cat_features)
        x, y = train_pool, eval_pool

        # train
        print('CatBoost is training')
        self.model.fit(x, eval_set=y, plot=False)
        
        # predict
        preds = self.model.predict_proba(x_valid[FE.features])[:, 1]
        acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_valid, preds)
        print(f"Valid Accuracy : {acc} Valid Auc : {auc}")

        # inference
        write_path = os.path.join(args.output_dir, "catboost.csv")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for id, p in enumerate(preds):
                w.write("{},{}\n".format(id, p))


class LGBM:

    def __init__(self, args):
        self.args = args
        self.model = LGBMClassifier(
            objective= 'binary',
            max_depth = 8,
            min_gain_to_split= 0.4,
            num_leaves= 100,
            num_iteration=1000,
            boost_from_average=False,
            metric = 'auc'
    
        )


    def train(self, args, y_train, x_train, y_valid, x_valid, cat_features):
        
        # train
        print('LGBM is training')
        self.model.fit(x_train[FE.features], y_train,early_stopping_rounds=100,eval_metric="logloss",eval_set=[(x_valid[FE.features], y_valid)])
        
        # predict
        preds = self.model.predict_proba(x_valid[FE.features])[:, 1]
        acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_valid, preds)
        print(f"Valid Accuracy : {acc} Valid Auc : {auc}")

        #inference
        write_path = os.path.join(args.output_dir, "lgbm.csv")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(write_path, "w", encoding="utf8") as w:
            w.write("id,prediction\n")
            for id, p in enumerate(preds):
                w.write("{},{}\n".format(id, p))
