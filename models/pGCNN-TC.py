import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
from tools import Dataset,loss_func_adjusted,test_epoch
from torch.utils import data
import random
import time
from torch.nn import init
from groupy.gconv.pytorch_gconv.splitgconv2d import P4ConvZ2, P4ConvP4

def weights_init(m):
    if isinstance(m, P4ConvZ2) or isinstance(m,P4ConvP4):
        init.normal_(m.weight.data,0,0.01)
        init.constant_(m.bias.data,0.1)
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0,0.01)
        m.bias.data.zero_()

class Net(nn.Module):
    def __init__(self,args):
        super(Net, self).__init__()
        self.conv2d_0 = P4ConvZ2(2, 16, 4, stride=2,padding=1) 
        self.conv2d_1 = P4ConvP4(16, 32, 3, stride=2, padding=1)
        self.batch_norm_1 = nn.BatchNorm3d(32)
        self.conv2d_2 = P4ConvP4(32, 64, 3, stride=2, padding=1) 
        self.conv2d_3 = P4ConvP4(64, 128, 3, stride=2, padding=1) 
        self.batch_norm_2 = nn.BatchNorm3d(128)
        self.linear_0 = nn.Linear(4*4*128*4+args.extra_features, 256)
        self.batch_norm_3 = nn.BatchNorm1d(256)
        self.linear_1 = nn.Linear(256, 128)
        self.linear_2 = nn.Linear(128, 1)

    def forward(self, train_img,extra_features=''):
        x = F.relu(self.conv2d_0(train_img))
        x = F.relu(self.conv2d_1(x))
        x = self.batch_norm_1(x)
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        x = self.batch_norm_2(x)

        xb = x.contiguous().view(-1, 4*4*128*4)
        if extra_features!='':
            xb = torch.cat((xb,extra_features),1)
        xb = F.relu(self.linear_0(xb))
        xb= self.batch_norm_3(xb) 
        xb = F.relu(self.linear_1(xb))
        xb = self.linear_2(xb)
        return xb

class BlendNet(nn.Module):
    def __init__(self, args, max_rotated_sita, blend_num,rotate,randcrop):
        super(BlendNet, self).__init__()
        self.net = Net(args).to(args.device)
        self.MAX_ROTATED_SITA = max_rotated_sita
        self.blend_num = blend_num
        self.device = args.device
        self.args = args
        self.rotate = rotate
        self.transform = transforms.RandomRotation(self.MAX_ROTATED_SITA)
        self.randcrop = randcrop 
    def forward(self, train_img,extra_features=''):
        pred = torch.zeros((self.blend_num, train_img.shape[0])).to(self.device)
        for i in range(self.blend_num):
            with torch.no_grad():
                train_x = train_img.permute(0,3,1,2)
                if self.rotate == 0:
                    train_x = train_x
                elif self.rotate == 1:
                    train_x = self.transform(train_x)
                elif self.rotate == 2:
                    train_x = transforms.functional.rotate(train_x,i*(360/self.blend_num))
                    
                if self.randcrop:
                    randnum_1 = round(65*0.2*random.random()-6.5)
                    randnum_2 = round(65*0.2*random.random()-6.5)
                    train_x = train_x[:,:,18+randnum_1:83+randnum_1,18+randnum_2:83+randnum_2]
                else:
                    train_x = train_x[:,:,18:83,18:83]
            pred[i] = self.net(train_x,extra_features).squeeze(-1)
        return pred

def train(args, model, optimizer, scheduler):

    train_set = Dataset(args,args.trainset_xpath, args.trainset_ypath,args.FeaTrainset_xpath)
    train_loader = data.DataLoader(train_set, batch_size = args.batch_size, shuffle = True)
    device = args.device
    logtxt = open(args.save_path +"/log.txt",'w')
    logtxt.write(str(args)+"\n")
    for epoch in range(args.epochs+1):
        start_time = time.time()
        model.train()
        for xx,yy,extra_features in train_loader:
            xx = xx.to(device)
            yy = yy.to(device) 
            if extra_features[0]!='':
                extra_features = extra_features.to(device)    
            else:
                extra_features = ''   
            optimizer.zero_grad()
            output = model(xx,extra_features)
            loss = loss_func_adjusted(output, yy,args)
            L1_reg = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if  'weight' in name:
                    L1_reg = L1_reg + torch.norm(param, 1)
            loss = loss + 5e-4 * L1_reg
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % args.log_interval == 0:
            print(f'Train Epoch: {epoch}, duration: {round(time.time()-start_time)}')
            eval(args,model.state_dict(),epoch,logtxt,rotationMethod=0,blend_num_whentest=1)
            eval(args,model.state_dict(),epoch,logtxt,rotationMethod=2,blend_num_whentest=10)
            test(args,model.state_dict(),epoch,logtxt,rotationMethod=0,blend_num_whentest=1)
            test(args,model.state_dict(),epoch,logtxt,rotationMethod=2,blend_num_whentest=10)
        if epoch > 500 and epoch % args.save_interval == 0:
                torch.save(model.state_dict(), args.save_path +"/epoch_"+str(epoch)+".pt")
    logtxt.close()

def test(args,model_state,epoch,logtxt,rotationMethod,blend_num_whentest):        
    torch.cuda.empty_cache()
    model = BlendNet(args, max_rotated_sita=180, blend_num=blend_num_whentest,rotate=rotationMethod,randcrop=False).to(args.device)
    model.load_state_dict(model_state)
    model.eval()
    eval_set = Dataset(args,args.testset_xpath, args.testset_ypath,args.FeaTestset_xpath)
    eval_loader = data.DataLoader(eval_set, batch_size = args.batch_size, shuffle = False)
    rmse,mae,preds,trues = test_epoch(args.device,eval_loader, model)    
    print(" test","rmse:",rmse,"mae:",mae)
    logtxt.write(" test rmse:"+str(rmse)+" mae:"+str(mae)+'\n')

def eval(args,model_state,epoch,logtxt,rotationMethod,blend_num_whentest):        
    torch.cuda.empty_cache()
    model = BlendNet(args, max_rotated_sita=180, blend_num=blend_num_whentest,rotate=rotationMethod,randcrop=False).to(args.device)
    model.load_state_dict(model_state)
    model.eval()
    eval_set = Dataset(args,args.evalset_xpath, args.evalset_ypath,args.FeaEvlset_xpath)
    eval_loader = data.DataLoader(eval_set, batch_size = args.batch_size, shuffle = False)
    rmse,mae,preds,trues = test_epoch(args.device,eval_loader, model)    
    print(" eval","rmse:",rmse,"mae:",mae)
    logtxt.write(" eval rmse:"+str(rmse)+" mae:"+str(mae)+'\n')

def main(args):
    model = BlendNet(args, max_rotated_sita=180, blend_num=args.blend_num,rotate=args.rotationMethod,randcrop=False).to(args.device)
    model.apply(weights_init)
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr,weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
    train(args, model, optimizer, scheduler)

if __name__=="__main__":
    parser =  argparse.ArgumentParser()
    parser.add_argument("--log_interval", default=1)
    pparser.add_argument("-Tx", "--trainset_xpath", default="", help="the trainning set x file path")
    parser.add_argument("-Ty", "--trainset_ypath", default="", help="the trainning set y file path")
    parser.add_argument("-Tex", "--testset_xpath", default="", help="the testing set x file path")
    parser.add_argument("-Tey", "--testset_ypath", default="", help="the testing set y file path")
    parser.add_argument("-Evlx", "--evalset_xpath", default="", help="the eval set x file path")
    parser.add_argument("-Evly", "--evalset_ypath", default="", help="the eval set y file path")
    parser.add_argument("--FeaTrainset_xpath", default="", help="Auxiliary features for training ")
    parser.add_argument("--FeaEvlset_xpath", default="", help="Auxiliary features for evalution")
    parser.add_argument("--FeaTestset_xpath", default="", help="Auxiliary features for testing")
    parser.add_argument("-B", "--batch_size", default=64)   
    parser.add_argument("-E", "--epochs",type=int, default=650, help="epochs for trainning")
    parser.add_argument("--blend_num",type=int,default=1)
    parser.add_argument("--save_interval",type=int,default=10)
    parser.add_argument("--rotationMethod",type=int,help="0 no roration, 1 is random rotation, 2 is blend_10 rotation")
    parser.add_argument("--gpu",type=int)
    parser.add_argument("--extra_features",type=int,default=0,help="number of auxiliary features used")
    parser.add_argument("--name",type=int,default=0)
    parser.add_argument("--range_low",type=int,default=45)
    parser.add_argument("--range_high",type=int,default=55)
    parser.add_argument("--AdjustL_coef",type=float,default=1.15)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    args.lr = 5e-3
    args.gamma = 0.90
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists("../results/"+os.path.basename(__file__).split('.')[0]+'/'+ str(args.gpu)+'_'+str(args.name)) :
        os.makedirs("../results/"+os.path.basename(__file__).split('.')[0]+'/'+ str(args.gpu)+'_'+str(args.name))
    args.save_path = "../results/"+os.path.basename(__file__).split('.')[0]+'/'+ str(args.gpu)+'_'+str(args.name)
    main(args)
