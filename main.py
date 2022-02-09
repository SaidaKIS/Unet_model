from cProfile import label
import dataset
import losses
import model
import train
import utils
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import pickle
import pandas as pd
from glob import glob
from torchsummary import summary

#change this in general 
#/Users/smdiazcas/miniconda/envs/pyUnet/lib/python3.9/site-packages/torch/storage.py
#class Unpickler(pickle.Unpickler):
#    def find_class(self, module, name):
#        if module == 'torch.storage' and name == '_load_from_bytes':
#            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#        else: return super().find_class(module, name)

def flatten(t):
    return [item for sublist in t for item in sublist]

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    bin_classes = ['Intergranular lane', 'Uniform-shape granules', 'Granules with dots', 'Granules with lanes',
                   'Complex-shape granules']

    #Parameters
    root = 'data/Masks_S_v4/Train/' # Raw full IMaX maps (6 for training and 1 for validate)
    l = 15000 # Submaps dataset size 
    size_box = 128 # size of each submap
    channels = 1
    N_EPOCHS = 200 
    BACH_SIZE = 32  
    loss = 'IoU' # 'CrossEntropy', 'FocalLoss', 'mIoU'
    save_model = True
    bilinear = False # Unet upsampling mechanisim is Traspose convolution
    model_summary = False
    lr = 1e-3
    dropout = False

    prop=pd.DataFrame(columns=[0, 1, 2, 3, 4], index=np.arange(0,2000))
    data=dataset.segDataset(root,l=2000, s=size_box)
    centre = []
    for i in range(2000):
        img, mask, ind, c = data[i]
        values, counts = np.unique(mask, return_counts=True)
        prop.loc[i, values] = np.array(counts/sum(counts))
        if ind == 0:
            centre.append(c)
        if i % 200 == 0:
            print(i)
            ax1 = plt.subplot(121)
            ax1.imshow(img[0])
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax2 = plt.subplot(122)
            ax2.imshow(mask)
            ax2.set_xticks([])
            ax2.set_yticks([])
            plt.tight_layout()
            plt.show()
    print(prop.mean())
    
    file_list = sorted(glob(root+'*.npz'))
    file = np.load(file_list[0])
    mask = file['cmask_map'].astype(np.float32)
    c = np.array(centre)
    utils.test_centers(mask, c[:,0], c[:,1])

    ##Train a model
    #train.run(root, l, size_box, channels, N_EPOCHS, BACH_SIZE, loss, lr = lr, scale=8,
    #    save_model=True, bilinear=False, model_summary=False, dropout=dropout)

    #Test model
    #Initial summary
    #model_unet = model.UNet(n_channels=1, n_classes=5, scale=8, bilinear=bilinear, dropout=dropout).to(device)
    #summary(model_unet, (channels, 128, 128))

    # Generate a prediction 
    #model_test1 = torch.load('../New_results/NewGT_Jan2022/Augmentation/unet_epoch_13_0.50860_IoU_non_Dropout_valaug.pt', map_location=torch.device(device))
    #file = 'data/Masks_S_v3/Train/Mask_data_Frame_0.npz'
#
    ##smap_f0, cmask_map_f0, total, total0, ls=utils.model_eval(file, model_test1, device, size_box)
    #smap_f0, cmask_map_f0, total, total0, ls=utils.model_eval_full(file, model_test1, device)
    #print(ls)
##
    ##utils.probability_maps(smap_f0[0], total[0], bin_classes)
    #utils.comparative_maps(smap_f0[0], cmask_map_f0[0], total0[0], bin_classes, save=True) 
    #imax_save = '/Users/smdiazcas/Documents/Phd/Research/NN_granulation/contmaps.sav'
    #utils.test_Imax(imax_save, model_test1, bin_classes)
   
    #Training information
    #with open ('../New_results/NewGT_Jan2022/Augmentation/Train_params_2022-02-07_20_04_54_FocalLoss-g10.npy', 'rb') as f:
    #    training_info = np.load(f, allow_pickle=True)
    #    metrics = np.load(f, allow_pickle=True)
    #    h_train_metrics = np.load(f, allow_pickle=True)
    #    h_val_metrics = np.load(f, allow_pickle=True)
####
    #print(training_info)
    #utils.metrics_plots(metrics, Title='Model: Loss Focal Loss SRS non-Dropout')
##
    #h_lt=[]
    #h_lv=[]
    #h_at=[]
    #h_av=[]
    #for i in range(5):
    #    h_lt.append(h_train_metrics[i,0,:])
    #    h_at.append(h_train_metrics[i,1,:])
    #    h_lv.append(h_val_metrics[i,0,:])
    #    h_av.append(h_val_metrics[i,1,:])
#
    #fig, ax =plt.subplots(nrows=2,ncols=2)
    #ax[0][0].hist(h_lt, bins=10)
    #ax[0][1].hist(h_at, bins=10)
    #ax[1][0].hist(h_lv, bins=10)
    #ax[1][1].hist(h_av, bins=10)
    #ax[0][0].set_title('Loss Training')
    ##ax[0][0].legend(prop={'size': 13})
    #ax[0][1].set_title('Acc Training')
    #ax[1][0].set_title('Loss Validation')
    #ax[1][1].set_title('Acc Validation')
    #ax[1][0].set_xlabel('Values')
    #ax[1][1].set_xlabel('Values')
    #ax[0][0].set_ylabel('Counts/Dataset elements')
    #ax[1][0].set_ylabel('Counts/Dataset elements')
#
    #plt.show()
##