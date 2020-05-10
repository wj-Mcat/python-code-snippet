import torch
import torch.nn.functional as F
from torch import nn

class SlotAttention(nn.Module):
    """
    拿来主义
    最简单的Attention机制
    """
    def __init__(self, n_features=64):
        super(SlotAttention, self).__init__()
        self.attention = nn.Linear(n_features, n_features)

    def forward(self, x):
        """
        :param x: hidden states of LSTM (batch_size, seq_len, hidden_size)
        :return: slot attention vector of size (batch_size, seq_len, hidden_size)

        attention = softmax(x * linear(x)) * x

        """

        weights = self.attention(x)  # (batch_size, seq_len, hidden_size) - temporary weight
        weights = torch.matmul(weights, torch.transpose(x, 1, 2))  # (batch_size, hidden_size, hidden_size) - att matrix
        weights = F.softmax(weights, dim=2)
        output = torch.matmul(weights, x)
        return output


class BertSelfAttention(nn.Module):

    """
    此代码是huggingface官方Transformers库中的示例代码
    """
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            # 最终的hidden_size是由多个Self-Attention拼接一起而成，故取余是需要等于0的
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x): 
        # (batch_size, sequence_length , hidden_size)
        # -> (batch_size, sequence_length, num_attention_heads, attention_head_size)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        # -> (batch_size, num_attention_head, attention_head_size)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # Q q_x
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs

def bert_example():
    """
    Bert的示例代码
    """
    from transformers import BertModel, BertTokenizer
    import torch

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',force_download=True)
    model = BertModel.from_pretrained('bert-base-uncased',force_download=True)
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)

    last_hidden_states = outputs[0] 


