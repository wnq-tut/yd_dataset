import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

'''
class MyLSTM(nn.Module):
    def __init__(self, n_feature, hidden_size, lstm_num_layers, lstm_dropout, l2_lambda, if_sigmoid, if_bidirection):
        super(MyLSTM, self).__init__()
        self.bn = nn.BatchNorm1d(n_feature)
        self.change_d_model = nn.Linear(in_features=n_feature, out_features=512)
        self.lstm = nn.LSTM(input_size=n_feature, hidden_size=hidden_size, num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=if_bidirection)
        self.linear = nn.Linear(in_features=hidden_size, out_features=1)
        self.linear_bidirection = nn.Linear(in_features=hidden_size*2, out_features=1)
        self.hidden_size = hidden_size
        self.l2_lambda = l2_lambda
        self.if_sigmoid = if_sigmoid
        self.if_bidirection = if_bidirection
    
    def forward(self, inputs, lengths):
        padded_inputs = inputs.to(torch.float).cuda()
        max_length = padded_inputs.shape[1]
        # inputs = inputs.permute(0, 2, 1)
        # inputs = self.bn(inputs)
        # inputs = inputs.permute(0, 2, 1)
    
        packed_inputs = nn.utils.rnn.pack_padded_sequence(padded_inputs, lengths, batch_first=True, enforce_sorted=False)
        # the shape of h_last: (num_layers * num_directions, batch_size, hidden_size)

        packed_outputs, _ = self.lstm(packed_inputs)
        # outputs, (h_last, c_last) = self.lstm(feats)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=max_length)
        # print(output.shape) # [batch_size, num_time_step, hidden_size * if_bi]
        forward_outputs  = output[range(len(output)), lengths - 1, :self.hidden_size]
        # print(forward_outputs.shape) # [batch_size, hidden_size]
        if self.if_bidirection:
            backward_outputs = output[:, 0, self.hidden_size:]
            # print(backward_outputs.shape) # [batch_size, hidden_size]
            output_combined = torch.cat([forward_outputs, backward_outputs], dim=1)
        # when single direction, h_last[-1] is output of the last layer.
        # but when bidirection is true, maybe need to use h_last[-1] and h_last[-2] which from different directions.
        # combine them and change the layer with shape of input is (hidden_size * 2)
        # h_last_combined = torch.cat((h_last[-1], h_last[-2]), dim=1)
        final_output = self.linear_bidirection(output_combined) if self.if_bidirection else self.linear(forward_outputs)
        # output = torch.mean(output, dim=-1)
        if self.if_sigmoid:
            final_output = torch.sigmoid(final_output) * 5
        return final_output

    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg
'''

class MyLSTM(nn.Module):
    def __init__(self, n_feature, d_model, hidden_size, lstm_num_layers, lstm_dropout, l2_lambda, if_sigmoid, if_bidirection, if_change_d_model):
        super(MyLSTM, self).__init__()
        self.bn = nn.BatchNorm1d(n_feature)
        self.change_d_model = nn.Linear(in_features=n_feature, out_features=d_model)
        input_size = d_model if if_change_d_model else n_feature
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=lstm_num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=if_bidirection)
        
        self.output_layer = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 1))
        self.output_layer_bi = nn.Sequential(nn.LayerNorm(hidden_size*2), nn.Linear(hidden_size*2, 1))

        self.hidden_size = hidden_size
        self.l2_lambda = l2_lambda
        self.if_sigmoid = if_sigmoid
        self.if_bidirection = if_bidirection
        self.if_change_d_model = if_change_d_model
    
    def forward(self, inputs, lengths):
        inputs = inputs.to(torch.float).cuda()
        max_length = inputs.shape[1]
        if self.if_change_d_model:
            inputs = self.change_d_model(inputs)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        # the shape of h_last: (num_layers * num_directions, batch_size, hidden_size)
        output, _ = self.lstm(inputs)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=max_length)
        forward_outputs  = output[range(len(output)), lengths - 1, :self.hidden_size]
        if self.if_bidirection:
            backward_outputs = output[:, 0, self.hidden_size:]
            bi_output = torch.cat([forward_outputs, backward_outputs], dim=1)
            output = self.output_layer_bi(bi_output)
        else:
            output = self.output_layer(forward_outputs)
        if self.if_sigmoid:
            output = torch.sigmoid(output) * 5
        return output

    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg

class OneDConvForScoring(nn.Module):
    def __init__(self, n_feature, batch_size, num_time_steps, l2_lambda=0):
        super(OneDConvForScoring, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=64, kernel_size=3, dilation=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Dynamically calculate the input dimensions of the fully connected layer
        input_shape = (batch_size, n_feature, num_time_steps)  # Assumed input size
        conv_output_size = self._get_conv_output(input_shape)
        
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        
        self.l2_lambda = l2_lambda
        
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(shape) # (32, 23, 499)
            output = self.pool(self.relu(self.bn1(self.conv1(input)))) # (32, 64, 248)
            output = self.pool(self.relu(self.bn2(self.conv2(output))))  # # (32, 128, 122)
            return torch.flatten(output, 1).size(1)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg
    
    def forward(self, inputs, lengths):
        # input shape of conv should be like (batch_size, n_feature, num_time_steps), which is diferent from lstm so permute it.
        x = inputs.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class OneDConvForClassification(nn.Module):
    def __init__(self, n_feature, batch_size, num_time_steps, l2_lambda):
        super(OneDConvForClassification, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_feature, out_channels=64, kernel_size=3, dilation=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Dynamically calculate the input dimensions of the fully connected layer
        input_shape = (batch_size, n_feature, num_time_steps)  # Assumed input size
        conv_output_size = self._get_conv_output(input_shape)
        
        # for type
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 5)
        # for level
        self.fc3 = nn.Linear(conv_output_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, 3)
        
        self.l2_lambda = l2_lambda
        
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(shape) # (32, 23, 499)
            output = self.pool(self.relu(self.bn1(self.conv1(input)))) # (32, 64, 248)
            output = self.pool(self.relu(self.bn2(self.conv2(output))))  # # (32, 128, 122)
            return torch.flatten(output, 1).size(1)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg
    
    def forward(self, inputs, lengths):
        # input shape of conv should be like (batch_size, n_feature, num_time_steps), which is diferent from lstm so permute it.
        x = inputs.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)
        x_type = self.fc1(x)
        x_type = self.dropout(x_type)
        x_type = self.fc2(x_type)
        x_level = self.fc3(x)
        x_level = self.dropout(x_level)
        x_level = self.fc4(x_level)
        return x_type, x_level
    
from torch.autograd import Variable
import math
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    



from typing import Optional, Any, Union, Callable

from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import _get_activation_fn


class CustomTransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False, if_need_weights: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation
        
        self.if_need_weights = if_need_weights
        self.attn_weights = None

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            is_causal: If specified, applies a causal mask as src_mask.
              Default: ``False``.
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )


        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        # Here the len of x depends on the setting of the need_weights parameter
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True, is_causal=is_causal)
        if self.if_need_weights:
            self.attn_weights = x[1]
        return self.dropout1(x[0])

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
    
    

    
class TransformerEncoderForScoring(nn.Module):
    def __init__(self, n_feature, d_model, n_head, max_len, num_encoder_layers=6, l2_lambda=0, use_checkpoint=True):
        super(TransformerEncoderForScoring, self).__init__()
        
        # The default value is false. It is only manually changed to true during the testing phase and when attention visualization needs to be generated. Note: Change it to false during training
        # self.get_atten_weights = False
        self.get_atten_weights = True
        
        # Add attention weights list
        self.all_attention_weights = []
        
        self.d_model = d_model
        self.l2_lambda = l2_lambda
        self.use_checkpoint = use_checkpoint
        
        self.change_d_model = nn.Linear(n_feature, d_model)
        
        self.position_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer Encoder
        encoder_layer = CustomTransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, if_need_weights=self.get_atten_weights)
        # encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        # self.transformer = nn.Transformer(d_model=d_model, nhead=n_head, num_encoder_layers=num_encoder_layers, dim_feedforward=d_model*4, batch_first=False)
        
        # Pooling Layer
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layer
        self.fc = nn.Linear(d_model, 1)

    @autocast()
    def forward(self, src, src_key_padding_mask):
        src = self.change_d_model(src)
        # Adding positional encoding
        src = src + self.position_encoder(src)
        
        # Through the Transformer encoder
        if self.use_checkpoint:
            transformer_output = checkpoint(self.encode, src, src_key_padding_mask)
        else:
            transformer_output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        # Pooling
        pooled_output = self.avg_pool(transformer_output.permute(1, 2, 0)).squeeze(-1)
        # Fully connected layer
        output = self.fc(pooled_output)
        output = torch.sigmoid(output) * 5
        return output
    
    # def encode(self, src, src_key_padding_mask):
    #     return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
    
    def encode(self, src, src_key_padding_mask):
        if self.get_atten_weights:
            # reset attention weights list
            single_data_attn = []
            for layer in self.transformer_encoder.layers:
                src = layer(src, src_key_padding_mask=src_key_padding_mask)
                single_data_attn.append(layer.attn_weights)
            self.all_attention_weights.append(single_data_attn)
            return src
        else:
            return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg