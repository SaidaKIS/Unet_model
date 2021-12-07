import dataset
import losses
import model
import train
import utils
import torch
import numpy as np

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    bin_classes = ['Intergranular lane', 'Granules with dots', 'Granules with lanes',
                   'Complex-shape granules', 'Normal-shape granules']

    model_test1 = torch.load('model_params/unet_epoch_35_0.49074_CE_TC.pt', map_location=torch.device(device))
    f = 'data/Masks_S_v2/Mask_data_Frame_0.npz'
    smap_f0, cmask_map_f0, total0, l=utils.model_eval(f, model_test1, device)

    l = np.load('model_params/Training_params_FL_TC_40_epocs.npy')
    utils.metrics_plots(l)
    utils.comparative_maps(smap_f0, cmask_map_f0, total0, bin_classes)   