import argparse
import json
import os
import copy
import time
from datetime import datetime
import logging
from tqdm import *

from torch.utils.tensorboard import SummaryWriter

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv


def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)

    data_prefix = './data/' + func_args.market + '/'
    matrix_path = data_prefix + func_args.relation_file

    start_time = datetime.now().strftime('%m%d_%H%M%S')  # 콜론 대신 언더스코어 사용
    if func_args.mode == 'train':
        PREFIX = 'outputs/'
        PREFIX = os.path.join(PREFIX, start_time)
        img_dir = os.path.join(PREFIX, 'img_file')
        save_dir = os.path.join(PREFIX, 'log_file')
        model_save_dir = os.path.join(PREFIX, 'model_file')

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(img_dir):
            os.makedirs(img_dir)
        if not os.path.isdir(model_save_dir):
            os.mkdir(model_save_dir)

        hyper = copy.deepcopy(func_args.__dict__)
        print(hyper)

        hyper['device'] = 'cuda' if hyper['device'] == torch.device('cuda') else 'cpu'
        json_str = json.dumps(hyper, indent=4)

        with open(os.path.join(save_dir, 'hyper.json'), 'w') as json_file:
            json_file.write(json_str)

        writer = SummaryWriter(save_dir)
        writer.add_text('hyper_setting', str(hyper))

        logger = logging.getLogger()
        logger.setLevel('INFO')
        BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('WARNING')
        fhlr = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
        fhlr.setFormatter(formatter)
        logger.addHandler(chlr)
        logger.addHandler(fhlr)

        if func_args.market == 'DJIA':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            assert stocks_data.shape[0] == rate_of_return.shape[0] and stocks_data.shape[2] == rate_of_return.shape[1], 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            test_idx = 7328
            allow_short = True
        elif func_args.market == 'HSI':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            assert stocks_data.shape[0] == rate_of_return.shape[0] and stocks_data.shape[2] == rate_of_return.shape[1], 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            test_idx = 4211
            allow_short = True

        elif func_args.market == 'CSI100':
            stocks_data = np.load(data_prefix + 'stocks_data.npy')
            rate_of_return = np.load(data_prefix + 'ror.npy')
            market_history = np.load(data_prefix + 'market_data.npy')
            assert stocks_data.shape[0] == rate_of_return.shape[0] and stocks_data.shape[2] == rate_of_return.shape[1], 'file size error'
            A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
            test_idx = 1944
            allow_short = False

        env = PortfolioEnv(assets_data=stocks_data, market_data=market_history, rtns_data=rate_of_return,
                           in_features=func_args.in_features, val_idx=test_idx, test_idx=test_idx,
                           batch_size=func_args.batch_size, window_len=func_args.window_len, trade_len=func_args.trade_len,
                           max_steps=func_args.max_steps, mode=func_args.mode, norm_type=func_args.norm_type,
                           allow_short=allow_short)

        supports = [A]
        actor = RLActor(supports, func_args).to(func_args.device)
        agent = RLAgent(env, actor, func_args)

        mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))
        try:
            max_cr = 0
            for epoch in range(func_args.epochs):
                epoch_return = 0
                for mini_batch in range(mini_batch_num):
                    episode_return, avg_rho, avg_mdd = agent.train_episode()
                    epoch_return += episode_return
                print(f'Epoch {epoch + 1}/{func_args.epochs}, Return: {epoch_return}')
                if epoch_return > max_cr:
                    max_cr = epoch_return
                    agent.save_model(model_save_dir)
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            agent.save_model(model_save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false')
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false')
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--market', type=str, default='DJIA')  # Add market argument
    args = parser.parse_args()

    # Load hyperparameters from config file
    if args.config:
        with open(args.config, 'r') as f:
            config_args = json.load(f)
        args_dict = vars(args)
        args_dict.update(config_args)
    else:
        args_dict = vars(args)

    # Initialize ConfigParser with dictionary
    config = ConfigParser(args_dict)
    run(config)