import os, sys, random, time, json, torch, kaldiio
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from models import *

from transformer_yd import Runner

class MyDataset(Dataset):
    def __init__(self, ark_path, if_norm, num_time_steps, padding=True):
        super(Dataset, self).__init__()
        # Make a file with the corresponding label of the path outside, use load_ark to process it here to get the data set, define len and getitem, and it's ok, you can pass it to dataloader for processing
        self.ark_path = ark_path
        self.if_norm = if_norm
        self.num_time_steps = num_time_steps
        self.feat_label_length_list = self.get_feat(self.ark_path, padding)
            
    def __len__(self):
        return len(self.feat_label_length_list)
    
    def __getitem__(self, idx):
        sample = self.feat_label_length_list[idx]
        return sample

    def get_feat(self, ark_path, padding):
        feat_label_length_list = []
       #Read the ark feature file and return a dictionary with the key as utt-id and the value as feature matrix (NumPy array)
        feats_dict = kaldiio.load_ark(ark_path)
        # Convert the feature data in the dictionary into label and feature lists
        feats_list = [[torch.tensor(feat), uttid] for uttid, feat in feats_dict]
        # feat_label: [tensor(num_time_steps, n_feature), uttid(str)]
        for feat_label in feats_list:
            length = feat_label[0].shape[0]
            if self.if_norm:
                feat_mean = torch.mean(feat_label[0], dim=0, keepdim=True)
                feat_std = torch.std(feat_label[0], dim=0, keepdim=True)
                feat_label[0] = (feat_label[0] - feat_mean) / feat_std
            # post paddding
            if padding:
                if length <= self.num_time_steps:
                    feat_label[0] = torch.nn.functional.pad(feat_label[0], (0, 0, 0, self.num_time_steps - length))
                else:
                    feat_label[0] = feat_label[0][:self.num_time_steps]
                feat_label.append(length)
            feat_label_length_list.append(feat_label)
        return feat_label_length_list
    
class Logger(object):
    def __init__(self, filename='log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Make sure to flush output to file
        self.terminal.flush()
        self.log.flush()
        
class ModelCheckpoint(object):
    def __init__(self, save_path, monitor='pcc', mode='max', which_dataset=None):
        self.save_path = save_path
        self.model_path = os.path.join(self.save_path, 'model.pth')
        self.monitor = monitor
        self.mode = mode
        self.best_value = -float('inf') if mode == 'max' else float('inf')

    def save_best(self, model, optimizer, feature, which_dataset=None, epoch=None, current_value=None):
        if self._is_improvement(current_value):
            print(f"{self.monitor} improved at epoch {epoch}: {current_value}. Saving model...\n\n")
            self.best_value = current_value
            for file in os.listdir(self.save_path):
                os.unlink(os.path.join(self.save_path, file))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_value': self.best_value,
                'dataset': which_dataset,
                'feature': feature,
                'epoch': epoch
            }, self.model_path)
            return True
        return False

    def _is_improvement(self, current_value):
        return (current_value > self.best_value) if self.mode == 'max' else (current_value < self.best_value)

class ModelEvaluator(object):
    def __init__(self):
        pass

    def evaluate_single_value(self, y_pred, y_true):
        values_matrix = torch.stack([y_pred, y_true])
        # Compute the covariance matrix between two tensors. This matrix is ​​a symmetric matrix with a shape of (x, x), where x is the number of tensors concatenated above, that is, the number of rows.
        # cov_matrix = torch.cov(values_matrix)
        # # cov_matrix[i ,j] represents the covariance of the i-th tensor and the j-th tensor, which is equivalent to cov_matrix[j, i].
        # cov = cov_matrix[0, 1]
        # Pearson's coefficient measures the degree of linear correlation between two variables.
        corr_matrix = torch.corrcoef(values_matrix)
        pearson = corr_matrix[0, 1]
        mse = torch.nn.functional.mse_loss(y_pred, y_true)
        mae = torch.mean(torch.abs(y_pred - y_true))
        print(f"mse: {mse:.2f}\nmae: {mae:.2f}\npearson: {pearson:.2f}")
        return mse, mae, pearson
            
    def evaluate_classification(self, save_path, y_true, y_pred, class_labels, task='task'):
        '''
        evaluate model
        Note: If the confusion matrix heat map only displays the first row of values, please downgrade the matplotlib version to 3.7.2
        '''
        # if y_true and y_pred are PyTorch Tensor, and maybe on the GPU, use this to convert them:
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        # Calculate basic indicators
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0, labels=np.unique(y_true))

        # Creating a confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)

        # Plotting the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap="BuPu", cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'{save_path}/Confusion_Matrix_of_{task}.png')
        plt.clf()
        
        # Create a DataFrame for the metrics
        metrics_data = {
            "Accuracy": [accuracy],
            "Precision": [precision],
            "Recall": [recall],
            "F1 Score": [f1]
        }
        metrics_df = pd.DataFrame(metrics_data)
        # Plotting the DataFrame as a table and saving as an image
        fig, ax = plt.subplots(figsize=(6, 1))  # Adjust the size as needed
        ax.axis('tight')
        ax.axis('off')
        ax.table(cellText=metrics_df.values, colLabels=metrics_df.columns, loc='center')
        plt.savefig(f"{save_path}/Metrics_Table_of_{task}.png")
        plt.close()
        
    def draw_curve(self, save_path, data, title, data_label='', x_label='epoch', y_label='loss', size=(10 ,5), fontsize=16):
        plt.figure(figsize=size)
        plt.plot(data, label=data_label)
        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(y_label, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.title(title)
        # Add legend
        # plt.legend()
        # Grid lines
        plt.grid(True)
        plt.savefig(f'{save_path}/{title}.png')
        plt.close()
        
    def draw_scatter(self, save_path, y_true, y_pred, scatter_random=False, name='points', size=(10 ,10), fontsize=16):
        if scatter_random:
            for i in range(len(y_true)):
                y_true[i] = y_true[i] + 0.1 * random.uniform(-1, 1)
            for i in range(len(y_pred)):
                y_pred[i] = y_pred[i] + 0.1 * random.uniform(-1, 1)
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        plt.figure(figsize=size)
        # Draw a scatter plot
        plt.scatter(y_true, y_pred, s=10)
        # Draw a straight line (e.g., y=x)
        # p.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)])
        plt.plot([0, 5], [0, 5])
        # Add tags and title
        # p.title(f'pcc = {pearson}\nmse = {mse}')
        plt.xlabel('Label value', fontsize=fontsize)
        plt.ylabel('Predicted value', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.savefig(f'{save_path}/scatter_{name}.png')
        plt.close()
        
    def plot_attention_focus(self, save_path, attn_weights, group_size=10):
        # Create a directory to save the heatmap
        save_path = os.path.join(save_path, 'attention_focus')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Iterate over each piece of data
        for idx, data in enumerate(attn_weights):
            # The shape of data is (6, 1, 2000, 2000)
            for layer_idx, layer_data in enumerate(data):
                # Extract the attention weights of layer , removing the batch dimension
                attention = layer_data[0]

                # Average the attention in dimension 0 (output time step), the shape of attention_avg is (2000, )
                attention_avg = torch.mean(attention, dim=0).cpu().numpy()
                
                # Average every group_size time steps
                num_groups = attention_avg.shape[0] // group_size
                attention_compressed = attention_avg[:num_groups * group_size].reshape(num_groups, group_size).mean(axis=1)
                attention_compressed = np.repeat(attention_compressed, group_size)

                
                # Set x-axis ticks and labels
                x_ticks = np.arange(0, num_groups * group_size, 100)
                x_labels = [str(i) for i in range(0, num_groups * group_size, 100)]  # 生成相应的标签
                
                 # Normalized
                # attention_compressed = (attention_compressed - np.min(attention_compressed)) / (np.max(attention_compressed) - np.min(attention_compressed))
                
                plt.figure(figsize=(15, 2))
                plt.imshow(attention_compressed[np.newaxis, :], aspect='auto', cmap='Reds', interpolation='bicubic')
                plt.title(f'Data {idx + 1} Layer {layer_idx + 1} Attention Focus')
                plt.xlabel('Input Time Step')
                plt.xticks(ticks=x_ticks, labels=x_labels)  # Set x-axis ticks and labels
                plt.yticks([])  # Remove y-axis ticks
                plt.ylabel('Attention Weight')
                plt.colorbar()

                # Save Heatmap
                plt.savefig(f'{save_path}/data_{idx + 1}_layer_{layer_idx + 1}_average.png')
                plt.close()
                
                # Set x- and y-axis tick labels
                x_ticks = np.arange(0, attention.shape[1], 100)
                x_labels = [str(i) for i in range(0, attention.shape[1], 100)]
                y_ticks = np.arange(0, attention.shape[0], 100)
                y_labels = [str(i) for i in range(0, attention.shape[0], 100)]
                
                # Plotting a heat map
                plt.figure(figsize=(15, 10))
                plt.imshow(attention.cpu().numpy(), aspect='auto', cmap='Reds', interpolation='nearest')
                plt.title(f'Data {idx + 1} Layer {layer_idx + 1} Attention Focus')
                plt.xlabel('Input Time Step')
                plt.xticks(ticks=x_ticks, labels=x_labels)  # Set x-axis ticks and labels
                plt.ylabel('Output Time Step')
                plt.yticks(ticks=y_ticks, labels=y_labels)  # Set the y-axis ticks and labels
                plt.colorbar()

                # Save Heatmap
                plt.savefig(f'{save_path}/data_{idx + 1}_layer_{layer_idx + 1}_full.png')
                plt.close()
            # break
        
        
class TestModel(object):
    def __init__(self, test_folder='test_model', pth_file_name='model.pth', param_dict_file_name='param_dict.json'):
        
        self.folder_path = os.path.join(os.getcwd(), test_folder)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
            print('created directory, put files into it.')
            exit(1)
        self.model_path = os.path.join(self.folder_path, pth_file_name)
        self.param_dict_path = os.path.join(self.folder_path, param_dict_file_name)
        with open(self.param_dict_path, 'r') as f:
            self.param_dict = json.load(f)
        self.evaluator = ModelEvaluator()

    def load(self, Model, Dataset, DataLoader):
        model_name = self.param_dict.get('model_name')
        if model_name is None:
            model = Model(**self.param_dict["model_dict"]).cuda()
        if model_name == 'transformer':
            model = TransformerEncoderForScoring(**self.param_dict["model_dict"]).cuda()
        if model_name == 'lstm':
            model = MyLSTM(**self.param_dict["model_dict"]).cuda()
        state_dict = torch.load(self.model_path)
        # print(checkpoint)
        model.load_state_dict(state_dict['model_state_dict'])
        ark_name = f'{state_dict["feature"]}.ark'
        
        # Special treatment is provided here
        feat_folder = state_dict["dataset"]
        if feat_folder in ['total', 'mis', 'smooth']:
            feat_folder = 'feat'
        elif feat_folder in ['a', 'c']:
            feat_folder = 'feat_' + state_dict["dataset"]
        else:
            feat_folder = 'feat'
        
        ark_path = os.path.join(os.getcwd(), feat_folder, 'test', ark_name)
        model.eval()
        dataset_test = Dataset(ark_path, **self.param_dict["dataset_dict"])
        data_loader_test = DataLoader(dataset_test, **self.param_dict["dataloader_dict"])

        # Modify this according to the situation
        with torch.no_grad():
            if isinstance(model, TransformerEncoderForScoring):
                run = Runner(model=model, optimizer=None, loss_fun=None, scaler=None, which_dataset=state_dict["dataset"])
                _, scores_all, labels_all = run(data_loader=data_loader_test, train=False, test=True)
                if model.get_atten_weights:
                    self.evaluator.plot_attention_focus(self.folder_path, model.all_attention_weights)
                    '''
            if isinstance(model, MyLSTM):
                from lstm_yd import run as lstm_run
                _, scores_all, labels_all = lstm_run(model=model, data_loader=data_loader_test, optimizer=None, loss_fun=None, label_name=state_dict["dataset"], train=False, test=True)
                '''
        mse, mae, pearson = self.evaluator.evaluate_single_value(scores_all, labels_all)
        self.evaluator.draw_scatter(self.folder_path, labels_all, scores_all, name='test')

        with open(f'{self.folder_path}/log.txt', "w") as f:
            f.write(f'mse:{mse}\nmae:{mae}\npcc:{pearson}\npredictions:\n{scores_all}\nlabels:\n{labels_all}\n\n\n{self.param_dict.get("parameter_dict")}')