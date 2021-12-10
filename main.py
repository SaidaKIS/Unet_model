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

    model_test1 = torch.load('../New_results/unet_epoch_45_0.51774_IoU_128x128.pt', map_location=torch.device(device))
    f = 'data/Masks_C/Mask_data_Frame_0.npz'
    #f = 'data/Masks_C/Mask_data_Frame_112.npz'
    smap_f0, cmask_map_f0, total0, ls=utils.model_eval(f, model_test1, device, 128)
    print(ls)

    l = np.load('../New_results/Training_params_IoU_128x128_50_epocs.npy')
    #utils.metrics_plots(l)
    utils.comparative_maps(smap_f0, cmask_map_f0, total0, bin_classes, save=True)   