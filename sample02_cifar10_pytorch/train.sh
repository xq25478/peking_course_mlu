set -e
set -x

export MLU_VISIBLE_DEVICES=0

source /torch/venv3/pytorch/bin/activate

python train.py
