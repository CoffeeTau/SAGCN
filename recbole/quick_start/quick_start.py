# @Time   : 2020/10/6, 2022/7/18
# @Author : Shanlei Mu, Lei Wang
# @Email  : slmu@ruc.edu.cn, zxcptss@gmail.com

# UPDATE:
# @Time   : 2022/7/8, 2022/07/10, 2022/07/13
# @Author : Zhen Tian, Junjie Zhang, Gaowei Zhang
# @Email  : chenyuwuxinn@gmail.com, zjj001128@163.com, zgw15630559577@163.com

"""
recbole.quick_start
########################
"""
import logging
import torch
from logging import getLogger

import sys


import pickle
import pandas as pd
from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
    save_split_dataloaders,
    load_split_dataloaders,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
)


def run_recbole(
    model=None, dataset=None, config_file_list=None, config_dict=None, args=None, saved=True
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified datasets

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): datasets name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])

    # ADD SOME MYSELF INDEX
    config["distance"] = args.distance
    config["old_new_dir"] = args.old_new_dir
    config["is_mean"] = args.is_mean
    config["alphaM"] = args.alphaM
    config["betaM"] = args.betaM
    config["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # logger initialization
    init_logger(config)
    logger = getLogger()

    file_handler = logging.FileHandler("run_result/log/" + args.run_topic + ".txt")
    logger.addHandler(file_handler)

    logger.info(sys.argv)
    logger.info(config)

    # datasets filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # datasets splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # users = []
    # items = []
    # for idx, batch in enumerate(train_data):
    #     users.append(batch["user_id"]-1)
    #     items.append(batch["item_id"]-1)
    # users = torch.hstack(users).numpy()
    # items = torch.hstack(items).numpy()
    # df = pd.DataFrame({"user": users, "item":items})
    # df = df.sort_values(by=["user", "item"])
    # df.to_csv("/data/guojiayan/GitHub/GDE/datasets/citeulike/train_sparse.csv", index=False)
    
    # users = []
    # items = []
    # for idx, batch in enumerate(valid_data):
    #     # print(batch[1])
    #     # print(batch[2])
    #     # print(batch[3])
    #     # break
    #     # print(batch[1])
    #     # print(batch[2])
    #     # print(batch[3])
    #     users.append(batch[2])
    #     items.append(batch[3])
    # users = torch.hstack(users).numpy()
    # items = torch.hstack(items).numpy()
    # df = pd.DataFrame({"user": users, "item":items})
    # df = df.sort_values(by=["user", "item"])
    # df.to_csv("/data/guojiayan/GitHub/GDE/datasets/citeulike/test_sparse.csv", index=False)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)

    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    run_topic = args.run_topic
    save_to_excel(best_valid_result, test_result, run_topic)

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    return {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def save_to_excel(best_valid_result, test_result, run_topic):
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet['A1'] = "index"
    sheet['B1'] = "score_best_valid"
    sheet['C1'] = "score_test_result"
    sheet.append(['recall@10', best_valid_result['recall@10'], test_result['recall@10']])
    sheet.append(['recall@20', best_valid_result['recall@20'], test_result['recall@20']])
    sheet.append(['recall@50', best_valid_result['recall@50'], test_result['recall@50']])
    sheet.append(['ndcg@10', best_valid_result['ndcg@10'], test_result['ndcg@10']])
    sheet.append(['ndcg@20', best_valid_result['ndcg@20'], test_result['ndcg@20']])
    sheet.append(['ndcg@50', best_valid_result['ndcg@50'], test_result['ndcg@50']])
    name = "run_result/excel/"+run_topic + ".xlsx"
    workbook.save(name)

def run_recboles(rank, *args):
    ip, port, world_size, nproc, offset = args[3:]
    args = args[:3]
    run_recbole(
        *args,
        config_dict={
            "local_rank": rank,
            "world_size": world_size,
            "ip": ip,
            "port": port,
            "nproc": nproc,
            "offset": offset,
        },
    )


def objective_function(config_dict=None, config_file_list=None, saved=True):
    r"""The default objective_function used in HyperTuning

    Args:
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
    """

    config = Config(config_dict=config_dict, config_file_list=config_file_list)
    init_seed(config["seed"], config["reproducibility"])
    logger = getLogger()
    for hdlr in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(hdlr)
    init_logger(config)
    logging.basicConfig(level=logging.ERROR)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"], config["reproducibility"])
    model_name = config["model"]
    model = get_model(model_name)(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, verbose=False, saved=saved
    )
    test_result = trainer.evaluate(test_data, load_best_model=saved)

    tune.report(**test_result)
    return {
        "model": model_name,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def load_data_and_model(model_file):
    r"""Load filtered datasets, split dataloaders and saved model.

    Args:
        model_file (str): The path of saved model file.

    Returns:
        tuple:
            - config (Config): An instance object of Config, which record parameter information in :attr:`model_file`.
            - model (AbstractRecommender): The model load from :attr:`model_file`.
            - datasets (datasets): The filtered datasets.
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    import torch

    checkpoint = torch.load(model_file)
    config = checkpoint["config"]
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.load_other_parameter(checkpoint.get("other_parameter"))

    return config, model, dataset, train_data, valid_data, test_data
