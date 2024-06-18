from torch.utils.data import DataLoader
from generic import MyDataset, TestModel
from models import TransformerEncoderForScoring, MyLSTM

test = TestModel()
model_dict = test.param_dict.get('model_dict')
if model_dict.get('max_len'):
    model = TransformerEncoderForScoring
else:
    model = MyLSTM
test.load(Model=model, Dataset=MyDataset, DataLoader=DataLoader)