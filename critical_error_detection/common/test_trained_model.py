from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel
from examples.sentence_level.wmt_2020.en_zh.monotransquest_config import monotransquest_config
import torch
MODEL_TYPE = "xlmroberta"
model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=1, use_cuda=torch.cuda.is_available(), args=monotransquest_config)
source = "Agreed any changes from 12 million to 9.4 will be reverted from now on.-CAYA"
translation = "从现在起 , 所有 更改 将 由 1200 万 改为 9.4 亿 。 - CAYA"
print(f"source: {source}")
print(f"translation: {translation}")
predictions, raw_outputs = model.predict([[source, translation]])
print(predictions, raw_outputs)