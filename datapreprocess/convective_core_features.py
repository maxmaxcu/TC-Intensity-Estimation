import numpy as np

def get_convective_core(data_x):
    data_x_1_0 = np.roll(data_x,1,0) 
    data_x_1_0[0] = float('inf')
    data_x_1_1 = np.roll(data_x,1,1)
    data_x_1_0[:,0] = float('inf')
    data_x_n1_0 = np.roll(data_x,-1,0)
    data_x_1_0[-1] = float('inf')
    data_x_n1_1 = np.roll(data_x,-1,1)
    data_x_1_0[:,-1] = float('inf')
    data_x = np.where((data_x>data_x_1_0) | (data_x > data_x_1_1)| (data_x >  data_x_n1_0) | (data_x >  data_x_n1_1),0,data_x)
    data_x = np.where(data_x>253,0,data_x)
    data_x = np.where((5.8/4*((data_x_n1_0+data_x_1_0-2*data_x)/3.1 + (data_x_1_1+data_x_n1_1-2*data_x)/8) > np.exp(0.0826*(data_x-217))) &(data_x>1),data_x,0)
    return data_x

## generate the convective cores image in npy format
def create_dataset(data_x_path='',data_xsave_path=''):
    data_x = np.load(data_x_path)[:,:,:,0]
    data_x_convective_core = np.zeros(data_x.shape)
    for i in range(data_x.shape[0]):
        data_x_convective_core[i] = get_convective_core(data_x[i]) 
    np.save(data_xsave_path,data_x_convective_core)

## generate the 8 auxiliary features from the created dataset using above function
def get_convective_core_features(datapath='',savePath=''):
    convcores = np.load(datapath)[:,31:70,31:70]
    features = np.zeros((convcores.shape[0],8))
    exceptions = 0
    success = 0
    for i in range(features.shape[0]):
        try: 
            features[i,0] = np.sum(np.where(convcores[i]>0,1,0)) ## N
            if features[i,0] == 0:
                features[i] = 0
            else:
                success += 1
                features[i,1] = convcores[i].max() ## T_max
                features[i,2] = np.where(convcores[i]>0,convcores[i],float('inf')).min() ## T_min
                features[i,3] = np.sum(convcores[i])/features[i,0] ## T_mean
                features[i,4] = features[i,1] - features[i,2] ## T_dif
                D = np.zeros(int(features[i,0]))
                D_locs = np.where(convcores[i]>0)
                for j in range(D.shape[0]):
                    D[j] = ((D_locs[0][j]-19)**2+(D_locs[1][j]-19)**2)**0.5
                features[i,5] = D.max() # D_max
                features[i,6] = D.min() # D_min
                features[i,7] = D.mean() # D_mean
        except:
            exceptions +=1
    print(convcores.shape,exceptions,success)
    np.save(savePath,features)

if __name__ == '__main__':
    create_dataset(data_x_path="",data_xsave_path="")
    get_convective_core_features(datapath='',savePath="")