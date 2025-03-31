import argparse
from recbole.quick_start import run_recbole

def main(args):
    
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=["configs/config_{}.yaml".format(args.dataset.lower())], args=args)
    
if __name__ == "__main__":

    model_name = "SAGCN"
    data_name = "gowalla"
    if model_name == "SAGCN":
        IfMine = True
    else:
        IfMine = False

    distance = "XiangDuiOuSi"
    old_new_dir = "new"
    is_mean =True
    alpha = 1.2
    beta = 0.001
    other = "Start Tuning - Distance"
    if IfMine:
        run_topic = model_name+"-"+data_name+" -dis_"+distance+" -alpha_"+str(alpha)+" -beta_"+str(beta)+" -direction_"+str(old_new_dir)+" -isMean_"+str(is_mean)+" -other_"+other
    else:
        run_topic = model_name + "-" + data_name

    print(run_topic)

    parser = argparse.ArgumentParser()

    parser.add_argument("--distance", default=distance, help="SA series use which distance")
    parser.add_argument("--old_new_dir", default=old_new_dir, help="decide the direction to log")
    parser.add_argument("--is_mean", default=is_mean, help="decide is or no to mean")
    parser.add_argument("--alphaM", default=alpha, help="the model's hyperparametric")
    parser.add_argument("--betaM", default=beta, help="the model's hyperparametric")
    parser.add_argument("--run_topic", default=run_topic, help="build excel with this name.")
    parser.add_argument("--model",   default=model_name, help="choose the model to use. ")
    parser.add_argument("--dataset", default=data_name, help="choose the datasets to use. ")
    args = parser.parse_args()
    main(args)
