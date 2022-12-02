import os

import torch
import wandb
from args import parse_args
from src.dataloader import Preprocess
from src.utils import setSeeds
from src.dataset import custom_train_test_split, make_dataset
from src.model import CatBoost
from src import trainer


def main(args):

    setSeeds(args.seed) 
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    cat_features = preprocess.get_features()

    ### cv 방법 고르기
    split = trainer.get_cv(args)
    train, test = split(train_data)

    # wandb.init(project="dkt", config=vars(args))

    y_train, x_train, y_valid, x_valid = make_dataset(train, test)
    model = trainer.get_model(args)
    model.train(args, y_train, x_train, y_valid, x_valid, cat_features)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
