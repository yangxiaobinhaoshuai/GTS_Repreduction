from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import argparse
import yaml
from model.pytorch.supervisor import GTSSupervisor
from lib.utils import load_graph_data
import uuid
import setproctitle
import torch

def main(args):
    setproctitle.setproctitle('yxb_{}'.format("gts_9968"))
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader)
        print("args.seq_len: ",supervisor_config['model']['seq_len'], " args.horizon: ",supervisor_config['model']['horizon'])
        save_adj_name = args.config_filename[11:-5]
        supervisor = GTSSupervisor(save_adj_name, temperature=args.temperature, **supervisor_config)
        supervisor.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='data/model/para_la.yaml', type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--temperature', default=0.5, type=float, help='temperature value for gumbel-softmax.')
    args = parser.parse_args()
    main(args)