import glob
import os
import random
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-o', type=str)
parser.add_argument('--dataset', type=str, default='ffhq', choices=['ffhq', 'celeba'])
parser.add_argument('-n', default='__', type=str, help='splits')
parser.add_argument('--sort', action='store_true')
parser.add_argument('--seed', default=np.random.randint(0, 10000), type=int)
args = parser.parse_args()

random.seed(args.seed)

if args.dataset == 'ffhq':
    total = 70000
elif args.dataset == 'celeba':
    total = 30000
else:
    raise NotImplementedError
pic_ids = [f'{i:05d}' for i in list(range(total))]
random.shuffle(pic_ids)

splits = []
for s in args.n.split('_'):
    if len(s) == 0:
        splits.append(0)
    else:
        splits.append(int(s))
print(f'splits: {splits}')
assert len(splits) == 3

num_train, num_val, num_test = splits[0], splits[1], splits[2]
if num_train == 0:
    num_train = total - num_val - num_test
assert num_train + num_val + num_test == total

train_ids = pic_ids[:num_train]
val_ids = pic_ids[num_train:num_train + num_val]
test_ids = pic_ids[num_train + num_val:]
if args.sort:
    train_ids = sorted(train_ids, reverse=False)
if args.sort:
    val_ids = sorted(val_ids, reverse=False)
if args.sort:
    test_ids = sorted(test_ids, reverse=False)

def save_split(ids, split):
    num = len(ids)
    ids_file = os.path.join(args.o, f'{split}.txt')
    with open(ids_file, mode='w') as ff:
        for pic_id in ids:
            ff.write(f'{pic_id}' + '\n')
    print(f'{split} set: {num}/{total}, will write to {ids_file}')

save_split(train_ids, 'train')
save_split(val_ids, 'val')
save_split(test_ids, 'test')
