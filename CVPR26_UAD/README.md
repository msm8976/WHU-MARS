#  The Unified Alignment and Discrimination (UAD) Framework

## Requirements

#### Datasets

- [WHU-MARS](https://github.com/msm8976/WHU-MARS#whu-mars-dataset)

#### Pre-trained model

- [jx_vit_base_p16_224](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)

#### Config

Please modify the dataset path and pre-trained weight path in `configs/*.yml` before running.


## Training

The released UAD setting uses 4 GPUs. On RTX 3090 GPUs, training on WHU-MARS-1000 takes about 2.5 hours.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=16666 train.py --config_file configs/UAD.yml MODEL.DIST_TRAIN True SOLVER.IMS_PER_BATCH 256 SOLVER.BASE_LR 0.032
```

Single-GPU training is also supported, but will take longer.

```bash
python train.py --config_file configs/UAD.yml
```

## Evaluation

Evaluate with 4 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --config_file configs/UAD.yml TEST.WEIGHT transformer_120.pth TEST.IMS_PER_BATCH 4096
```

Evaluate with a single GPU:

```bash
python test.py --config_file configs/UAD.yml TEST.WEIGHT transformer_120.pth
```

## Acknowledgement

- Our implementation is built on [TransReID](https://github.com/damo-cv/TransReID).

Many thanks to the authors of these excellent works!
