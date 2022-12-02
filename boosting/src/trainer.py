import math
import os

import torch
import wandb


from .model import CatBoost, LGBM
from .dataset import custom_train_test_split, custom_train_test_split_2


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "catboost":
        model = CatBoost(args)
    if args.model == "lgbm":
        model = LGBM(args)

    return model

def get_cv(args):
    """
    CV 방법 고르기
    """
    if args.cv == 'custom1':
        cv = custom_train_test_split
    if args.cv == 'custom2':
        cv = custom_train_test_split_2

    return cv


def save_checkpoint(state, model_dir, model_filename):
    print("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
