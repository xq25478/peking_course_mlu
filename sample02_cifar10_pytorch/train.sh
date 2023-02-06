set -e
set -x

#指定使用卡号为0的显卡
export MLU_VISIBLE_DEVICES=0

#激活虚拟环境
source /torch/venv3/pytorch/bin/activate

#执行训练脚本
python train.py
