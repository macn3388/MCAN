import os
import json
import time
import importlib
import argparse
from collections import OrderedDict
import torch
from dataset import TestDataset
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ckpt_path", type=str, default="None")
    parser.add_argument("--sample_dir", type=str, default="sample/")
    parser.add_argument("--sample_scale", type=int, default=2)
    parser.add_argument("--group", type=int, default=1)
    parser.add_argument("--shave", type=int, default=20)
    parser.add_argument("--sample_data_set", type=str, default="calculate_sets_xx/Set5")
    return parser.parse_args()


def save_image(tensor, filename):
    im = Image.fromarray(tensor)
    im.save(filename)


def sample(net, device, dataset, cfg):
    for step, (hr, lr, name) in enumerate(dataset):
        if "DIV2K" in dataset.name or True:
            t1 = time.time()

            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave

            # split large image to 4 patch to avoid OOM error
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(device)

            # run refine process in here!
            sr = net(lr_patch, cfg.sample_scale).data

            h, h_half, h_chop = h* cfg.sample_scale, h_half* cfg.sample_scale, h_chop* cfg.sample_scale
            w, w_half, w_chop = w* cfg.sample_scale, w_half* cfg.sample_scale, w_chop* cfg.sample_scale

            # merge splited patch images
            result = torch.FloatTensor(3, h, w).to(device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result

            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            # sr = result
            t2 = time.time()
        else:
            t1 = time.time()
            lr = lr.unsqueeze(0).to(device)
            sr = net(lr, cfg.sample_scale).detach().squeeze(0)
            lr = lr.squeeze(0)
            t2 = time.time()
        
        model_name = cfg.ckpt_path.split(".")[0].split("/")[-1]
        sr_dir = os.path.join(cfg.sample_dir,
                              model_name,
                              cfg.sample_data_set.split("/")[-1],
                              "x{}".format(cfg.sample_scale),
                              "SR")
        hr_dir = os.path.join(cfg.sample_dir,
                              model_name,
                              cfg.sample_data_set.split("/")[-1],
                              "x{}".format(cfg.sample_scale),
                              "HR")

        os.makedirs(sr_dir, exist_ok=True)
        os.makedirs(hr_dir, exist_ok=True)

        sr_im_path = os.path.join(sr_dir, "{}".format(name.replace("HR", "SR")))
        hr_im_path = os.path.join(hr_dir, "{}".format(name))

        save_image(sr, sr_im_path)
        save_image(hr, hr_im_path)
        print("Saved {} ({}x{} -> {}x{}, {:.3f}s)"
            .format(sr_im_path, lr.shape[1], lr.shape[2], sr.shape[1], sr.shape[2], t2-t1))

def main(cfg):
    model = importlib.import_module("model.{}".format(cfg.model)).Net
    print(json.dumps(vars(cfg), indent=4, sort_keys=True))

    if cfg.sample_scale > 0:
        net = model(scale=cfg.sample_scale, group=cfg.group)
    else:
        net= model(multi_scale=True, group=cfg.group)

    state_dict = torch.load(cfg.ckpt_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:] # remove "module."
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    if cfg.sample_scale > 0:
        dataset = TestDataset('./dataset/%s' % cfg.sample_data_set, cfg.sample_scale)
        sample(net, device, dataset, cfg)
    else:
        for s in range(2,5):
            cfg.sample_scale = s
            dataset = TestDataset('./dataset/%s' % cfg.sample_data_set, cfg.sample_scale)
            sample(net, device, dataset, cfg)


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
    print('done')
