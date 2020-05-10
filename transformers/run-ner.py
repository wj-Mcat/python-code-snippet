import argparse,glob,os,random,logging,torch
import numpy as np
from tqdm import tqdm,trange
from seqeval.metrics import f1_score,precision_score,recall_score

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader,SequentialSampler,RandomSampler,TensorDataset
from transformers import (
    WEIGHTS_NAME,
    AdamW,

    BertConfig,
    BertForTokenClassification,
    BertTokenizer,

    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,

    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,

    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,

    XLMConfig,
    XLMRobertaForTokenClassification,
    XLMTokenizer,

    get_linear_schedule_with_warmup
)