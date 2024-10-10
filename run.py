import argparse
from recbole.quick_start import run_recbole

def main(args):
    
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=["configs/config_{}.yaml".format(args.dataset.lower())], args=args)
    
if __name__ == "__main__":

    model_name = "SAGCN" #  LightGCN  DirectAU SAGCN  SADirectAU
    data_name = "citeulike" #  yelp2018  ml_1m   gowalla  citeulike(1,3是大数据集）
    if model_name == "SAGCN":
        IfMine = True # True  False
    else:
        IfMine = False
    # 上面第一个和第三个还没跑
    distance = "XiangDuiOuSi" # XiangDuiOuSi(现在用的)  JueDuiOuSi   YuXian   KLSanDu
    # is_change_old_new = False # 这个可以用公式来改变
    # is_change_old_new = True # 这个可以用公式来改变
    old_new_dir = "new"  # "old"
    is_mean = False
    alpha = 1.2  # 0.5  0.8  1  1.2   1.5   2  5
    beta = 1 # 0.001        # KLSanDu 20 100
    other = "重新复现-调参citeulike"  # "1个log" "两个log" "没有log" “”
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
    parser.add_argument("--dataset", default=data_name, help="choose the dataset to use. ")
    args = parser.parse_args()
    main(args)
