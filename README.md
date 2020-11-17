# BERT_with_Att_Flow
This is a model combining attention flow from [BiDAF paper](https://arxiv.org/abs/1611.01603) and pretrained BERT encoders for question answering task.

## List of files:

|Name|Description|
|---|---|
|``main.ipynb``| main notebook for model construction and training|
|``data_processing.py``| handler of SQuAD dataset|
|``bert_plus_bidaf.py``| definition of our model|
|``att_flow.py``| definition of the attention layer|
|``char_cnn.py``| definition of the CNN layer|
|``pred_layer.py``| definition of the prediction layer|
