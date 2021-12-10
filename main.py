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

    #Parameters
    root = 'data/Masks_C/'
    l = 100
    size_box = 96
    channels = 1
    N_EPOCHS = 2
    BACH_SIZE = 4
    loss = 'mIoU' # 'CrossEntropy', 'FocalLoss', 'mIoU'
    save_model = True
    bilinear = False
    model_summary = False
    lr = 1e-3

    #Train a model
    train.run(root, l, size_box, channels, N_EPOCHS, BACH_SIZE, loss, lr = lr, 
        save_model=True, bilinear=False, model_summary=False)

    #Test model
    # Generate a prediction 
    #model_test1 = torch.load('model_params/*.pt', map_location=torch.device(device))
    #file = 'data/Masks_C/Validate/Mask_data_Frame_76.npz'
    #smap_f0, cmask_map_f0, total0, ls=utils.model_eval(file, model_test1, device, size_box)
    #print(ls)
    #utils.comparative_maps(smap_f0, cmask_map_f0, total0, bin_classes, save=True)   

    # Training information
    #with open ('model_params/Train_params_2021-12-10 12:58:43.135519.npy', 'rb') as f:
    #    training_info = np.load(f, allow_pickle=True)
    #    metrics = np.load(f, allow_pickle=True)

    utils.metrics_plots(metrics)
