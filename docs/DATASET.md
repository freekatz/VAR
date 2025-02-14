## FFHQ 1024 (for train and validate)
download: https://drive.google.com/file/d/1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv/view?usp=drive_link
resize to 256 (optional):
```shell
python scripts/resize_ffhq_dataset.py -i <ffhq_1024_dir> -o <ffhq_256_dir> --size 256 --num_workers <cpu_numbers>
```
split dataset:
```shell
# train set 69500, val set 500
python scripts/split_dataset.py --dataset ffhq -o <ffhq_256_dir> -n '69500_500_' --sort
```

## CelebA HQ 1024 (for test)
download: https://www.kaggle.com/datasets/lamsimon/celebahq/data
split dataset:
```shell
# test set 160
python scripts/split_dataset.py --dataset ffhq -o <celeba_1024_dir> -n '_2_160' --sort
```

