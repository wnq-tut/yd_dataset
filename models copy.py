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
    
class TransformerEncoderForScoring(nn.Module):
    def __init__(self, n_feature, d_model, n_head, max_len, num_encoder_layers=6, l2_lambda=0, use_checkpoint=True):
        super(TransformerEncoderForScoring, self).__init__()
        
        self.d_model = d_model
        self.l2_lambda = l2_lambda
        self.use_checkpoint = use_checkpoint
        
        self.change_d_model = nn.Linear(n_feature, d_model)
        
        self.position_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4)
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
    
    def encode(self, src, src_key_padding_mask):
        return self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
    
    def l2_regularization(self):
        l2_reg = torch.tensor(0.0).cuda()
        for name, param in self.named_parameters():
            if 'weight' in name:
                l2_reg += torch.norm(param, p=2)
        return self.l2_lambda * l2_reg