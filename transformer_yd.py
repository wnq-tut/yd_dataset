from typing import Any
from generic import *
from models import *
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Runner(object):
    def __init__(self, model, optimizer, loss_fun, scaler, which_dataset=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss_fun
        self.scaler = scaler
        self.which_dataset = which_dataset
        
    def __call__(self, data_loader, train=True, test=False, *args: Any, **kwds: Any) -> Any:
        if train:
            self.model.train()
        else:
            self.model.eval()
        loss_all = 0.0
        scores_all = torch.tensor([]).cuda()
        labels_all = torch.tensor([]).cuda()
        for data in data_loader:
            feats, labels, lengths = self.process_data(data)
            padding_mask = self.create_padding_mask(feats, lengths)
            if train:
                # make the gradient zero before the backward
                self.optimizer.zero_grad()
            with autocast():
                # run model
                scores = self.model(feats.cuda(), padding_mask.cuda())
                # scores shape is (batch_size, 1), bur labels shape is(batch_size)，so we have to squeeze. if there's just one data in the last batch which means the shape is (1)，then restore
                scores = torch.squeeze(scores) if scores.shape[0] != 1 else scores.reshape(1)
                # compute loss
                loss = self.loss_fun(scores, labels) + self.model.l2_regularization() if test is not True else torch.tensor(0).cuda(0)
            if train:
                # backward
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            loss_all += loss.detach().item()
            scores_all = torch.cat((scores_all, scores), dim=0)
            labels_all = torch.cat((labels_all, labels), dim=0)
        return loss_all, scores_all, labels_all
    
    def process_data(self, data):
        label_class = {'mis': 1, 'smooth': 2, 'total': 3}
        # feats shape is (batch_size, num_time_steps, n_feature),labels shape is (batch_size)
        feats, uttids, lengths = data
        print(uttids)
        # process the uttids from ark file, format is uttname-label1-label2-label3, get the target label
        labels = [float(uttid.split('-')[label_class[self.which_dataset]]) for uttid in uttids]
        labels = torch.tensor(labels, dtype=torch.float16).cuda()
        feats = feats.permute(1, 0, 2)
        return feats, labels, lengths
    
    def create_padding_mask(self, feats, lengths):
        # create a range tensor which has the same shape as the input sequence
        max_len = feats.shape[0]
        range_tensor = torch.arange(max_len).expand(len(lengths), max_len)
        # create padding_mask，the padding will be set to True
        return range_tensor >= lengths.unsqueeze(1).float()

def save_param_dict(log_folder, n_feature, d_model, n_head, if_norm, max_len, batch_size, num_workers, shuffle=True, drop_last=False):
    param_dict = {
        "model_name": "transformer",
        "model_dict":{
            "n_feature": n_feature,
            "d_model": d_model,
            "n_head": n_head,
            "max_len": max_len
        },
        "dataset_dict":{
            "if_norm": if_norm,
            "num_time_steps": max_len
        },
        "dataloader_dict":{
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "drop_last": drop_last
        }
    }
    with open(os.path.join(log_folder, 'param_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(param_dict, f)
    return param_dict



        

def main(parameter_dict):
    torch.cuda.empty_cache()
    
    # the default config is for mfcc
    # number of time steps/squence length, which is derived by computing the longest sequence in data set
    num_time_steps = parameter_dict["num_time_steps"]
    # dimension of features
    n_feature = parameter_dict["mfcc"]["n_feature"]
    # hidden layer size
    hidden_size = parameter_dict["mfcc"]["hidden_size"]
    # batch size
    batch_size_train = parameter_dict["mfcc"]["batch_size_train"]
    batch_size_val = parameter_dict["mfcc"]["batch_size_val"]
    # number of epochs
    n_epoch = parameter_dict["n_epoch"]
    n_warm_up = int(n_epoch / 10)
    # learning rate
    lr = parameter_dict["lr"]
    # lstm layers number
    lstm_num_layers = parameter_dict["lstm_num_layers"]
    # L2 regularization
    l2_lambda = parameter_dict["l2_lambda"]
    # if use sigmoid
    if_sigmoid = parameter_dict["if_sigmoid"]
    lstm_dropout = parameter_dict["lstm_dropout"]
    loss_name = parameter_dict["loss_name"]
    optimizer_name = parameter_dict["optimizer_name"]
    # mfcc has been normalized when extracting by kaldi
    if_norm = parameter_dict["mfcc"]["if_norm"]
    # feature type
    feature = parameter_dict["feature"]
    # if make the scatter figure random
    scatter_random = parameter_dict["scatter_random"]
    # if ealy stopping
    if_early_stopping = parameter_dict["if_early_stopping"]
    # threshold of ealy stopping
    es_threshold = parameter_dict["es_threshold"]
    # if use bidirectional lstm
    if_bidirection = parameter_dict["if_bidirection"]
    # if use over resampling dataset
    use_over_resampling = parameter_dict["use_over_resampling"]
    # use which dataset
    dataset = parameter_dict["dataset"]
    # number of encoder layers for transformer
    num_encoder_layers = parameter_dict["num_encoder_layers"]
    # cut or pad the sequence to max length
    max_length = parameter_dict["max_length"]
    # if use checkpoint(a method to solve the oom problem)
    use_checkpoint = parameter_dict["use_checkpoint"]
    
    
    
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name == "NVIDIA GeForce MX350":
        device = 'son'
        num_workers = 0
        # os.chdir("D:/workspace/yd/")
    else:
        device = 'you'
        num_workers = parameter_dict["num_workers"]
        # os.chdir("/home/you/workspace/yd_dataset")
                
    feature_folder_dict = parameter_dict["feature_folder"]

    feature_folder_over = feature_folder_dict["over_resampling"]
    feature_folder = feature_folder_dict["feat"]
        
    if feature in parameter_dict:
        feature_params = parameter_dict[feature]
        n_feature = feature_params["n_feature"]
        hidden_size = feature_params["hidden_size"]
        batch_size_train = feature_params["batch_size_train"]
        batch_size_val = feature_params["batch_size_val"]
        if_norm = feature_params["if_norm"]
        # for transformer
        d_model = feature_params["d_model"]
        n_head = feature_params["n_head"]
        # name feature file like "{feature}_train.ark"
        if use_over_resampling:
            ark_path_train = f'{pwd}/{feature_folder_over}/{dataset}/{feature}.ark'
        else:
            ark_path_train = f'{pwd}/{feature_folder}/train/{feature}.ark'
        ark_path_val = f'{pwd}/{feature_folder}/val/{feature}.ark'
    else:
        raise ValueError(f"Unsupported feature type: {feature}")
                
    
    # generate dataset
    dataset_train = MyDataset(ark_path_train, if_norm, max_length)
    dataset_val = MyDataset(ark_path_val, if_norm, max_length)
        
    # generate dataloader
    # return an iterator.
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, drop_last=False)
    data_loader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False, num_workers=num_workers, drop_last=False)
    
    # for best performance
    checkpoint_best = ModelCheckpoint(save_path=best_path)
    
    evaluator = ModelEvaluator()
        
    # instantiate model
    # model = MyLSTM(n_feature, hidden_size, lstm_num_layers, lstm_dropout, l2_lambda, if_sigmoid, if_bidirection).cuda()
    model = TransformerEncoderForScoring(n_feature=n_feature, d_model=d_model, n_head= n_head, max_len=max_length, num_encoder_layers=num_encoder_layers, use_checkpoint=use_checkpoint)
    model = model.cuda()
        
    # define loss function
    # compute the average loss of each batch, can be changed to sum
    loss_fun = nn.MSELoss(reduction='mean').cuda()
    if loss_name == 'cross':
        loss_fun = nn.CrossEntropyLoss().cuda()
        
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
    scaler = GradScaler()

    # Create a ReduceLROnPlateau scheduler.
    # If there is no significant improvement in patience epochs, then lr = lr * factor
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    run = Runner(model, optimizer, loss_fun, scaler, which_dataset=dataset)
    
    # strat training
    loss_train_list = []
    loss_val_list = []
    pearson = 0
    
    for epoch in range(n_epoch):
        torch.cuda.empty_cache()
        # adjust learning rate dynamically
        # adjust_lr(optimizer, epoch, pearson)
        # make sure that don't require the atten_weight when training
        if model.get_atten_weight:
            print('model.get_atten_weight is true, please set it to false.')
            return 0
        print(f'epoch: {epoch+1}\nThe performance in the dataset_train:')
        loss_train, scores_train_all, labels_train_all = run(data_loader_train)
        loss_train_list.append(loss_train)
        print(f'The loss of train: {loss_train:.2f}')
        evaluator.evaluate_single_value(scores_train_all, labels_train_all)
        # validation
        print('The performance in the dataset_val:')
        loss_val, scores_val_all, labels_val_all = run(data_loader_val, train=False)
        loss_val_list.append(loss_val) 
        print(f"The loss of val: {loss_val:.2f}")
        mse, mae, pearson = evaluator.evaluate_single_value(scores_val_all, labels_val_all)
        print('\n')
        # adjust learning rate according to loss
        scheduler.step(loss_train)
        # save best 
        if checkpoint_best.save_best(model=model, optimizer=optimizer, feature=feature, which_dataset=dataset, epoch=epoch, current_value=pearson):
            save_param_dict(best_path, n_feature, d_model, n_head, if_norm, max_length, batch_size_val, num_workers)
            evaluator.draw_scatter(best_path, labels_val_all, scores_val_all, name='float')
            best_pearson = pearson
            best_scores_val_all = scores_val_all
            best = f'best performance:\nepoch: {epoch}\nmse: {mse}\nmae: {mae}\npearson: {pearson}\n'
        
        if if_early_stopping:
            if pearson>=es_threshold:
                print('early stopping')
                break
    
    evaluator.draw_curve(log_folder,loss_train_list, 'loss_train')
    evaluator.draw_curve(log_folder, loss_val_list, 'loss_val')

    print(f'\n\n\nThe hyperparameter list:\n{parameter_dict}')
    print(f'{labels_val_all}\n{best_scores_val_all}')
    print(best)
    
    return best_pearson
        
    
if __name__ == "__main__":
    argument = sys.argv[1:]
    if not torch.cuda.is_available():
        print("No GPU found")
        exit(1)
    pwd = os.getcwd()
    # load hyper-parameters file
    with open('parameter.json', 'r') as f:
        parameter_group = json.load(f)
        
    if not os.path.exists('log'):
        # create log folder
        os.makedirs('log')
    pwd = os.getcwd()
    os.chdir('./log')
    
    if type(parameter_group) is list:
        for parameter_dict in parameter_group:
            
            # get current time
            current_time = time.strftime('%Y-%m-%d_%H_%M', time.localtime())
            # create log_current_time folder
            log_folder_name = f'log_{current_time}'
            os.makedirs(log_folder_name, exist_ok=True)
            # change to log_current_time folder
            log_folder = os.path.join(pwd, 'log', log_folder_name)
            os.chdir(log_folder)
            best_path = os.path.join(log_folder, 'best')
            os.makedirs(best_path, exist_ok=True)
            # save original sys.stdout
            original_stdout = sys.stdout
            # open a file to write the output
            sys.stdout = Logger(stream=original_stdout)
            
            pcc = main(parameter_dict)
            
            # resrtore sys.stdout
            sys.stdout = original_stdout
            os.chdir(f'{pwd}/log')
            new_logger_folder = f'{log_folder}_{pcc:.3f}'
            os.rename(log_folder, new_logger_folder)
    else:
        print(f'parameter_group.json error:\n{parameter_group}')
        exit(1)