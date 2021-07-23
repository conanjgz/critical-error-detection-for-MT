CUDA_VISIBLE_DEVICES=1 python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 20 --seed 1000 --log ner.log -c config/enzh_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 20 --seed 1200 --log ner.log -c config/enzh_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 20 --seed 1500 --log ner.log -c config/enzh_ner.conf

CUDA_VISIBLE_DEVICES=1 python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 10 --seed 1000 --log ner.log -c config/enja_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 10 --seed 1200 --log ner.log -c config/enja_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 10 --seed 1500 --log ner.log -c config/enja_ner.conf

CUDA_VISIBLE_DEVICES=1 python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 20 --seed 1000 --log ner.log -c config/ende_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 20 --seed 1200 --log ner.log -c config/ende_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 20 --seed 1500 --log ner.log -c config/ende_ner.conf
