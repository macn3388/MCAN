import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
import skimage
from skimage import measure
import numpy as np

class Solver():
    def __init__(self, model, cfg):
        torch.backends.cudnn.benchmark = True
        self.cfg = cfg
        self.refiner = model(multi_scale=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = nn.L1Loss()
        self.refiner = self.refiner.to(self.device)
        self.loss_fn = self.loss_fn
        self.step = 0

    def psnr(self, im1, im2):
        def im2double(im):
            min_val, max_val = 0, 255
            out = (im.astype(np.float64) - min_val) / (max_val - min_val)
            return out

        im1 = im2double(im1)
        im2 = im2double(im2)
        psnr = skimage.measure.compare_psnr(im1, im2, data_range=1)
        return psnr

    def calculate(self):
        cfg = self.cfg
        self.load(cfg.ckpt_path)
        for scale in range(2, 5):
            if cfg.scale != 0 and cfg.scale != scale:
                continue
            tmp_scale = cfg.scale
            cfg.scale = scale
            rgb_psnr_list, rgb_avg_psnr = self.evaluate_metrics(["./dataset/%s" % cfg.test_data_set], cfg)
            print('######################## scale: %d ########################' % scale)
            for _dir in sorted(rgb_psnr_list.keys()):
                print('%16s rgb_psnr: %5f' % (_dir + '_rgb', rgb_psnr_list[_dir]))
            cfg.scale = tmp_scale

    def fit(self):
        cfg = self.cfg
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        self.train_data = TrainDataset(cfg.train_data_path,
                                       scale=cfg.scale,
                                       size=cfg.patch_size)
        self.train_loader = DataLoader(self.train_data,
                                       batch_size=cfg.batch_size,
                                       num_workers=1,
                                       shuffle=True, drop_last=True, pin_memory=True)
        self.optim = optim.Adam(filter(lambda p: p.requires_grad, self.refiner.parameters()), cfg.lr)
        refiner = nn.DataParallel(self.refiner, device_ids=range(cfg.num_gpu))
        learning_rate = cfg.lr
        while True:
            for inputs in self.train_loader:
                self.refiner.train()
                if cfg.scale > 0:
                    scale = cfg.scale
                    hr, lr = inputs[-1][0], inputs[-1][1]
                else:
                    scale = random.randint(2, 4)
                    hr, lr = inputs[scale-2][0], inputs[scale-2][1]
                hr = hr.to(self.device)
                lr = lr.to(self.device)
                sr = refiner(lr, scale)
                loss = self.loss_fn(sr, hr)
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.refiner.parameters(), cfg.clip)
                self.optim.step()
                learning_rate = self.decay_learning_rate(learning_rate)
                for param_group in self.optim.param_groups:
                    param_group["lr"] = learning_rate
                self.step += 1
                if self.step % cfg.print_interval == 0:
                    self.print_info(cfg, loss, learning_rate)
            if self.step > cfg.max_steps: break
            self.save(cfg.ckpt_dir, cfg.ckpt_name)

    def print_info(self, cfg, loss, lr):
        print('step: %d/%d, batch_size: %d, patch_size: %d, decay: %dk/%d chance, lr: %.2e, scale: %d, loss: %.5f'
              % (self.step, cfg.max_steps, cfg.batch_size, cfg.patch_size,
                 cfg.decay_interval/1000, cfg.decay_chance, lr, cfg.scale, loss.data.item()))

    def evaluate_metrics(self, test_list, cfg):
        rgb_ret = {}
        rgb_sum = 0.0
        for dir in test_list:
            dirname = dir.split("/")[-1]
            rgb_psnr = self.evaluate(dir, scale=cfg.scale)
            rgb_ret[dirname] = rgb_psnr
            rgb_sum += rgb_psnr
        return rgb_ret, rgb_sum/len(test_list)

    def evaluate(self, test_data_dir, scale=2):
        cfg = self.cfg
        rgb_mean_psnr = 0
        self.refiner.eval()

        test_data = TestDataset(test_data_dir, scale=scale)
        test_loader = DataLoader(test_data,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False)
        for step, inputs in enumerate(test_loader):
            hr = inputs[0].squeeze(0)
            lr = inputs[1].squeeze(0)

            h, w = lr.size()[1:]
            h_half, w_half = int(h/2), int(w/2)
            h_chop, w_chop = h_half + cfg.shave, w_half + cfg.shave
            lr_patch = torch.FloatTensor(4, 3, h_chop, w_chop)
            lr_patch[0].copy_(lr[:, 0:h_chop, 0:w_chop])
            lr_patch[1].copy_(lr[:, 0:h_chop, w-w_chop:w])
            lr_patch[2].copy_(lr[:, h-h_chop:h, 0:w_chop])
            lr_patch[3].copy_(lr[:, h-h_chop:h, w-w_chop:w])
            lr_patch = lr_patch.to(self.device)
            sr = self.refiner(lr_patch, scale).data
            h, h_half, h_chop = h*scale, h_half*scale, h_chop*scale
            w, w_half, w_chop = w*scale, w_half*scale, w_chop*scale
            result = torch.FloatTensor(3, h, w).to(self.device)
            result[:, 0:h_half, 0:w_half].copy_(sr[0, :, 0:h_half, 0:w_half])
            result[:, 0:h_half, w_half:w].copy_(sr[1, :, 0:h_half, w_chop-w+w_half:w_chop])
            result[:, h_half:h, 0:w_half].copy_(sr[2, :, h_chop-h+h_half:h_chop, 0:w_half])
            result[:, h_half:h, w_half:w].copy_(sr[3, :, h_chop-h+h_half:h_chop, w_chop-w+w_half:w_chop])
            sr = result
            hr = hr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            sr = sr.cpu().mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            # evaluate PSNR
            # this evaluation is different to MATLAB version
            # we evaluate PSNR in RGB channel not Y in YCbCR
            bnd = scale
            im1 = hr[bnd:-bnd, bnd:-bnd]
            im2 = sr[bnd:-bnd, bnd:-bnd]
            rgb_mean_psnr += self.psnr(im1, im2) / len(test_data)
        return rgb_mean_psnr

    def load(self, path):
        self.refiner.load_state_dict(torch.load(path))
        print("Load pretrained {} model".format(path))

    def save(self, ckpt_dir, ckpt_name):
        save_path = os.path.join(
            ckpt_dir, "{}_{}.pth".format(ckpt_name, self.step))
        torch.save(self.refiner.state_dict(), save_path)

    def decay_learning_rate(self, now_lr):
        if self.cfg.decay_chance <=0:
            return now_lr
        lr = self.cfg.lr * (0.5 ** (self.step // self.cfg.decay_interval))
        if lr != now_lr:
            self.cfg.decay_chance -= 1
            self.cfg.decay_interval = 400000
        return lr




