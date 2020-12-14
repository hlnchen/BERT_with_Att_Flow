"""
This file is for plotting the attention weights in the BiDAF layer of a certain input context question pair.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import collections, time, spacy, copy, string, re, json, matplotlib
from layers.bert_plus_bidaf import BERT_plus_BiDAF
from eval_test import evaluate, compare
from utils import data_processing
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from training import SquadDataset

def getLen(token_list):
    for i, token in enumerate(token_list):
        if token == '[PAD]':
            return i
def attention_map(idx, eval_dataset, model):
    """ 
    Given an id in the eval_dataset, return the q2c and c2q attention 
    weights of the question and the context with all padding tokens removed.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    nlp = spacy.blank("en")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    model.to(device)

    # Get the attention weights
    model.eval()
    with torch.no_grad():
        input_ids = torch.unsqueeze(eval_dataset[idx]['input_ids'], dim=0).to(device)
        attention_mask = torch.unsqueeze(eval_dataset[idx]['attention_mask'], dim=0).to(device)
        outputs = model(input_ids, attention_mask)
    weight_c2q = outputs[3]
    weight_q2c = outputs[4]

    question_ids = input_ids[:,1:63].squeeze(0)
    context_ids = torch.cat((input_ids[:,0], input_ids[:,63:].squeeze(0)))
    question_tokens = tokenizer.convert_ids_to_tokens(question_ids)
    context_tokens = tokenizer.convert_ids_to_tokens(context_ids)
    
    question_len = getLen(question_tokens)
    context_len = getLen(context_tokens)
    question_tokens = question_tokens[0:question_len]
    context_tokens = context_tokens[0:context_len]

    weight_c2q_np = weight_c2q.squeeze(0).cpu().numpy()[0:context_len, 0:question_len]
    weight_c2q_np = weight_c2q_np / np.sum(weight_c2q_np, axis=1)[:,None]
    
    weight_q2c_np = weight_q2c.cpu().numpy()[:,0:context_len]
    weight_q2c_np = weight_q2c_np / np.sum(weight_q2c_np, axis=1)
    
    return weight_c2q_np, weight_q2c_np, question_tokens, context_tokens

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

if __name__ == "__main__":
    
    val_encodings = torch.load(r'D:\OneDrive\Courses\ECS289 NLP\val_encodings.pt')
    val_answer = torch.load(r'D:\OneDrive\Courses\ECS289 NLP\val_answer.pt')
    val_dataset = SquadDataset(val_encodings)

    model = BERT_plus_BiDAF(if_bidirectional=True,if_extra_modeling=True, if_attention_map=True)
    model.load_state_dict(torch.load(r'D:\OneDrive\Courses\ECS289 NLP\bert_bidaf_bidirectionalLSTM.pt'))

    idx = 0
    weight_c2q_np, weight_q2c_np, question_tokens, context_tokens = attention_map(idx, val_dataset, model)

    # c2q plot
    fig, ax = plt.subplots(figsize = (60,60))
    im, cbar = heatmap(weight_c2q_np.T, question_tokens, context_tokens, ax = ax, cmap = "GnBu", cbarlabel='Context to Question Attention Weights', aspect = 'auto')
    plt.savefig('C2Q_weights.jpeg', dpi=180, optimize=True)
    plt.show()

    # q2c plot
    fig, ax = plt.subplots(figsize = (60,20))
    im, cbar = heatmap(weight_q2c_np, ['Context'], context_tokens, ax = ax, cmap = "GnBu", cbarlabel='Question to Context Attention Weights', aspect = 'equal')
    plt.savefig('Q2C_weights.jpeg', dpi=180, optimize=True)
    plt.show()