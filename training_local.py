import os


config = "./configs/mini_mb_istft_vits2_base.json"
mpath = "./models/test"

os.system(f"python train.py -c {config} -m {mpath}")
