# Critical-Error-Detection-For-MT
## How to run:
In the project directory (either **CED_original** or **CED_modify_hidden_states**), run following command:
```
python main.py --log <log_name>.log -c config/<language_pair>.conf
```
Here, `<log_name>` is where you can specify the name of the log file of a training process. `<language_pair>` can be `enzh`, `enja`, `ende`, `encs` or other conf files in ***config*** directory.

Alternatively, if you want to directly modify one or two hyperparameters, you can use the following command to specify them in command line (e.g.):
```
python main.py --lr 2e-5 --warmup_ratio 0.3 --num_epochs 20 --log tox.log -c config/enzh_tox.conf
```

To cope with randomness and fluctuation, it is suggested to run the same set of hyperparameters (same conf file) three times with different seed each time, which can be set using `--seed`. Here is an example:
```
python main.py --log ner.log --seed 1200 -c config/enzh_ner.conf
```

If you have multiple GPUs and want to specify one GPU to train this model, you can add the `CUDA_VISIBLE_DEVICES=<GPU index>` at the beginning of your command:
```
CUDA_VISIBLE_DEVICES=1 python main.py --log ner.log --seed 1200 -c config/enzh_ner.conf
```

## Outputs
All training process will be recorded in the log file saved in a folder according to the date in **output/temp/**.

After a full training process, the tokenizer and the model with best MCC score will be saved to a folder named by date (e.g. "model_2021_07_21_11_39_40_enzh") in **output/temp/**. The predictions on test set will be save as **predictions.txt** in this folder as well. The path to this folder is given in the log file. 
