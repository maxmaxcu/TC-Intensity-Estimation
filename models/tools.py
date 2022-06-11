import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
import torch.nn as nn

class Dataset(data.Dataset):
    def __init__(self, args,direc_x,direc_y,direc_features,minV=0,maxV=200):

        dataset_x = torch.from_numpy(np.load(direc_x)).float() 
        dataset_y = torch.tensor(np.load(direc_y)).float()  
        if args.extra_features > 0:
            dataset_features = torch.from_numpy(np.load(direc_features)).float()
        else:
            dataset_features = ''
        dataset_x = dataset_x[dataset_y<=maxV,:,:,:]
        if args.extra_features > 0:
            dataset_features = dataset_features[dataset_y<=maxV]
        dataset_y = dataset_y[dataset_y<=maxV]
        dataset_x = dataset_x[minV < dataset_y,:,:,:]
        if args.extra_features > 0:
            dataset_features = dataset_features[minV < dataset_y]
        dataset_y = dataset_y[minV < dataset_y]
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.args = args
        if args.extra_features>0:
            self.dataset_features = dataset_features
        else:
            self.dataset_features = ""
    
    def __len__(self):
        return len(self.dataset_x)

    def __getitem__(self, index):
        data_x = self.dataset_x[index]
        data_y = self.dataset_y[index]
        if self.args.extra_features>0:
            dataset_features = self.dataset_features[index]
        else:
            dataset_features=""
        return data_x,data_y,dataset_features


def loss_func_mse(pred,gt):
    mean = torch.mean(pred,axis=0)
    return F.mse_loss(mean,gt)


class AdjustedRegressionLoss(nn.Module):
    def __init__(self,args):
        super(AdjustedRegressionLoss,self).__init__()
        self.range_low = args.range_low
        self.range_high = args.range_high
        self.AdjustL_coef = args.AdjustL_coef
    def forward(self,preds,labels):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loss_1 = torch.where((((labels>self.range_high)&(labels>preds)) | ((labels<self.range_low)&(labels<preds))),self.AdjustL_coef*torch.square(labels-preds),torch.square(labels-preds))
        loss = torch.where((((labels>self.range_high)&(labels<preds)) | ((labels<self.range_low)&(labels>preds))),(2-self.AdjustL_coef)*torch.square(labels-preds),loss_1)
        loss = torch.mean(loss)
        return loss

def loss_func_adjusted(pred,gt,args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = AdjustedRegressionLoss(args).to(device)
    return criterion(pred,gt)


def test_epoch(device,test_loader, model):
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy, extra_features in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)      
            if extra_features[0]!='':
                extra_features = extra_features.to(device)    
            else:
                extra_features = ''   
            output = model(xx,extra_features)
            preds.append(output.cpu().data.numpy())
            trues.append(yy.cpu().data.numpy())
        preds = np.concatenate(preds, axis = 1)  
        trues = np.concatenate(trues, axis = 0) 
        rmse_loss_avg = round((((preds.mean(axis=0) - trues)**2).mean())**0.5,2)
        mae_loss_avg = round(np.abs(preds.mean(axis=0)-trues).mean(),2)
        
    return rmse_loss_avg,mae_loss_avg,preds,trues
