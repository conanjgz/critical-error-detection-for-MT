CUDA_VISIBLE_DEVICES=1 python main.py --seed 1000 --log ner_4_types_with_sampler.log -c config/enzh_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1200 --log ner_4_types_with_sampler.log -c config/enzh_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1500 --log ner_4_types_with_sampler.log -c config/enzh_ner.conf

CUDA_VISIBLE_DEVICES=1 python main.py --seed 1000 --log ner_4_types_with_sampler.log -c config/enja_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1200 --log ner_4_types_with_sampler.log -c config/enja_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1500 --log ner_4_types_with_sampler.log -c config/enja_ner.conf

CUDA_VISIBLE_DEVICES=1 python main.py --seed 1000 --log ner_4_types_with_sampler.log -c config/ende_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1200 --log ner_4_types_with_sampler.log -c config/ende_ner.conf
CUDA_VISIBLE_DEVICES=1 python main.py --seed 1500 --log ner_4_types_with_sampler.log -c config/ende_ner.conf
