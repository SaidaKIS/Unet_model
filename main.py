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

#change this in general 
#/Users/smdiazcas/miniconda/envs/pyUnet/lib/python3.9/site-packages/torch/storage.py
#class Unpickler(pickle.Unpickler):
#    def find_class(self, module, name):
#        if module == 'torch.storage' and name == '_load_from_bytes':
#            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#        else: return super().find_class(module, name)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    bin_classes = ['Intergranular lane', 'Granules with dots', 'Granules with lanes',
                   'Complex-shape granules', 'Normal-shape granules']

    #Parameters
    root = 'data/Masks_S_v3/' # Raw full IMaX maps (6 for training and 1 for validate)
    l = 30000 # Submaps dataset size 
    size_box = 128 # size of each submap
    channels = 1
    N_EPOCHS = 200 
    BACH_SIZE = 32  
    loss = 'mIoU' # 'CrossEntropy', 'FocalLoss', 'mIoU'
    save_model = True
    bilinear = False # Unet upsampling mechanisim is Traspose convolution
    model_summary = False
    lr = 1e-3

    #prop=pd.DataFrame(columns=[0, 1, 2, 3, 4], index=np.arange(0,2000))
    #data=dataset.segDataset(root,l=2000, s=size_box)
    #centre = []
    #for i in range(2000):
    #    img, mask, ind, c = data[i]
    #    values, counts = np.unique(mask, return_counts=True)
    #    prop.loc[i, values] = np.array(counts/sum(counts))
    #    if ind == 0:
    #        centre.append(c)
    #    if i % 200 == 0:
    #        print(i)
    #        plt.imshow(mask)
    #        plt.show()
    #print(prop.mean())
    #
    #file_list = sorted(glob(root+'*.npz'))
    #file = np.load(file_list[0])
    #mask = file['cmask_map'].astype(np.float32)
    #c = np.array(centre)
    #utils.test_centers(mask, c[:,0], c[:,1])


    #Train a model
    train.run(root, l, size_box, channels, N_EPOCHS, BACH_SIZE, loss, lr = lr, 
        save_model=True, bilinear=False, model_summary=False)

    #Test model
    # Generate a prediction 
    #model_test1 = torch.load('../New_results/unet_epoch_19_0.60844.pt', map_location=torch.device(device))
    #file = 'data/Masks_C/Validate/Mask_data_Frame_76.npz'
    #smap_f0, cmask_map_f0, total, total0, ls=utils.model_eval(file, model_test1, device, size_box)
    #print(ls)
    #utils.probability_maps(smap_f0, total0, bin_classes)
    #utils.comparative_maps(smap_f0, cmask_map_f0, total, bin_classes, save=True) 
    #imax_save = '/dat/quest/QUEST_WP1/SUNRISE/2009_6_10/2009y_06m_10d_save/contmaps.sav'
    #utils.test_Imax(imax_save, model_test1, bin_classes)

    # Training information
    #with open ('../New_results/Train_params_2021-12-14_20_31_02.370241.npy', 'rb') as f:
    #    training_info = np.load(f, allow_pickle=True)
    #    metrics = np.load(f, allow_pickle=True)

    #print(training_info)
    #utils.metrics_plots(metrics)
