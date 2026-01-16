# Model training and sampling

We provide bash files [train_ddbm.sh](train_ddbm.sh) and [sample_ddbm.sh](sample_ddbm.sh) for model training and sampling. 
Simply set variables `DATASET_NAME` and `SCHEDULE_TYPE`:

- `DATASET_NAME` specifies which dataset to use. For each dataset, make sure to set the respective `DATA_DIR` variable in `args.sh` to your dataset path.
- `SCHEDULE_TYPE` denotes the noise schedule type. vp was used in this experiment.

## To train, run

CUDA_VISIBLE_DEVICES=1 bash train_ddbm.sh e2h vp

### To resume, set CKPT to your checkpoint, or it will automatically resume from your last checkpoint based on your experiment name.
CUDA_VISIBLE_DEVICES=1 bash train_ddbm.sh e2h vp $CKPT

## For inferring kinetic parameters from images, additional variables need to be set:

- `MODEL_PATH` is your checkpoint to be evaluated.
- `SPLIT` denotes which split you use for testing. Only `train` and `test` are supported.

To sample, run：
CUDA_VISIBLE_DEVICES=0 bash sample_ddbm.sh e2h vp model_170000.pt 0 1 test

## To finally infer the complete dynamic PET data, please run the program：

test_ours_fdg_zubal_head_sample3_img_k.py