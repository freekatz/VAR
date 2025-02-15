## Prepare
download vqvae pretrained weights: https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth
download var pretrained weights (d16): https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth

## Start

train/val/test
```shell
# batch size pre GPU is 11 (32G GPU), so `bs = 8*11 = 88`. if you have 32 GPUs, you can set bs = 32*11 = 352
torchrun --nproc_per_node=8 main.py --vae_path <vae_path> \
--dataset_name ffhq_blind --data_path <ffhq_256_dir> \
--dataset_name_test celeba_hq_blind --data_path_test <celeba_1024_dir> \
--exp_name test --bs 88 --workers 6 --img_size 256 --seed 2025 \
--tblr 0.001 --pretrain <var_path> --val_freq 10 \
--test_freq 10 --ep 500
```