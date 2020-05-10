from torch import nn
"""
1. 在batch模式下的数据训练过程中，文本长度的不一致处理是很常见的

pad_packed_sequence / pack_padded_sequence  这两个函数能

"""
# 比如在Intent的pad过程是需要使用填充值，而这个填充值在计算交叉熵时是需要忽略的。
# -100 就是忽略的值
def data_process():
    from torch.nn.utils.rnn import pad_sequence
    y_slot_train = pad_sequence(y_slot_train, batch_first=True,
                                padding_value=-100).to(torch.long)  # size: tensor(batch, max_seq_len, SLOTS)    
    cr_slot = nn.CrossEntropyLoss(ignore_index=-100)
    cr_intent = nn.CrossEntropyLoss()