from multiprocessing import cpu_count

SEED = 777
TEMP_DIRECTORY = "critical_error_detection/en_zh/temp/data"
RESULT_FILE = "result.tsv"
SUBMISSION_FILE = "predictions.txt"
RESULT_IMAGE = "result.jpg"
GOOGLE_DRIVE = False
DRIVE_FILE_ID = None
MODEL_TYPE = "xlmroberta"
MODEL_NAME = "xlm-roberta-large"

monotransquest_config = {
    'output_dir': 'critical_error_detection/en_zh/temp/outputs/',
    "best_model_dir": "critical_error_detection/en_zh/temp/outputs/best_model",
    'cache_dir': 'critical_error_detection/en_zh/temp/cache_dir/',

    'labels_list': ['NOT', 'ERR'],
    'labels_map': {'NOT': 0, 'ERR': 1},

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 64,
    'train_batch_size': 8, #32 64 128 
    'gradient_accumulation_steps': 1, #1-64
    'eval_batch_size': 4,
    'num_train_epochs': 10,
    'weight_decay': 0,
    'learning_rate': 2e-5, #1e-5 5e-5 max:1e-4 min:1e-6
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1, #0.2
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False, #try True, check XLMR

    'logging_steps': 300,
    'save_steps': 300,
    "no_cache": False,
    "no_save": False,
    "save_recent_only": False,
    'save_model_every_epoch': True,
    'n_fold': 1,
    'evaluate_during_training': True,
    "evaluate_during_training_silent": False,
    'evaluate_during_training_steps': 300,
    "evaluate_during_training_verbose": True,
    'use_cached_eval_features': False,
    "save_best_model": True,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,
    "save_optimizer_and_scheduler": True,

    'regression': False,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    "multiprocessing_chunksize": 500,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "early_stopping_delta": 0,
    "early_stopping_metric": "eval_loss",
    "early_stopping_metric_minimize": True,
    "early_stopping_consider_epochs": False,

    "manual_seed": SEED,

    "config": {},
    "local_rank": -1,
    "encoding": None,
}
