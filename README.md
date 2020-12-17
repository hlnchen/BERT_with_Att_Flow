# BERT_with_Att_Flow
This is a model combining attention flow from [BiDAF paper](https://arxiv.org/abs/1611.01603) and pretrained BERT encoders for question answering task.

## List of files:

|Name|Description|
|---|---|
|``training.py``|for training|
|``eval.py``|for evaluation on the dev set|
|``attention_map.py``|to visualize the attention matrices|
|``attention_map.ipynb``|an interactive version of the above|
|``error_analysis.ipynb``|to sample and categorize errors|
|``utils/data_processing.py``| handler of SQuAD dataset|
|``layers/bert_plus_bidaf.py``| definition of our model|
|``layers/att_flow.py``| definition of the attention layer|
|``layers/pred_layer.py``| definition of the prediction layer|

## Instruction:
Run ``training.py`` with inputs ``learning rate, batch size, number of epochs`` to train the model on the SQuAD v2.0 dataset.