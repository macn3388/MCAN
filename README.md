# MCAN

This code is based on https://github.com/nmhkahn/CARN-pytorch.

### Test Pretrained Models
```shell
$ python3 mcan/sample.py --model mcan-fast \
                         --sample_scale 0 \
                         --sample_dir ./sample \
                         --sample_data_set calculate_sets_x2/Set5 \
                         --ckpt_path checkpoint/MCAN-FAST.pth
```

Sampling the results of models. In the `--sample_scale` argument, [2, 3, 4] is for single-scale sampling and 0 for multi-scale sampling.

### Training Models
```shell
# for MCAN
$ python3 mcan/train.py --patch_size 64 \
                        --batch_size  64 \
                        --max_steps 1200000 \
                        --model mcan \
                        --train_data_path dataset/DIV2K_train_x234.h5 \
                        --num_gpu 1 \
                        --ckpt_name mcan \
                        --ckpt_dir checkpoint/mcan \
                        --decay_interval 400000 \
                        --lr 2e-4 \
                        --print_interval 10 \
                        --decay_chance 2
```

In the `--scale` argument, [2, 3, 4] is for single-scale training and 0 for multi-scale learning. `--decay_chance` represents the times of learning-rate decay throughout the training process.


### Calculating PSNR
```shell
$ python3 mcan/calculate.py --model mcan-fast \
                            --scale 2 \
                            --test_data_set calculate_sets_x2/Set5 \
                            --ckpt_path  checkpoint/MCAN-FAST.pth
```

