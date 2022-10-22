python -m prepocess.build_dicts
python -m prepocess.embd
python -m prepocess.user_embd
python -m prepocess.build_train --processes=10
python -m prepocess.build_dev --processes=10
python -m prepocess.build_test --processes=20
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=6 --epoch=8

CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=6 --epoch=8 --filenum=20