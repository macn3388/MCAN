import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import argparse
import importlib
from solver import Solver

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_path", type=str, default="None")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--test_data_set", type=str, default="calculate_sets_x2/Set14")
    parser.add_argument("--shave", type=int, default=20)
    return parser.parse_args()

def main(cfg):
    net = importlib.import_module("model.{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))
    solver = Solver(net, cfg)
    solver.calculate()
    print('Done')

if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
