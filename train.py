from collections import OrderedDict
import dataset
import losses
import model
import utils
import numpy as np
import sys
import torch
from torch import nn
from datetime import datetime
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run(root, l, size_boxes, channels, N_EPOCHS, BACH_SIZE, loss, lr =1e-3, 
        save_model=False, bilinear=False, model_summary=False):

    if loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss().to(device)
    if loss == 'FocalLoss':
        criterion = losses.FocalLoss(gamma=1).to(device)
    if loss == 'mIoU':
        criterion = losses.mIoULoss(n_classes=5).to(device)

    test_num = int(0.1 * l)
    data_train=dataset.segDataset(root+'Train/', l=l-test_num, s=size_boxes)
    data_test=dataset.segDataset(root+'Validate/', l=test_num, s=size_boxes)
    
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=BACH_SIZE, shuffle=True, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=BACH_SIZE, shuffle=False, num_workers=1)
    
    n_class = len(data_train.bin_classes)
    
    model_unet = model.UNet(n_channels=channels, n_classes=n_class, bilinear=bilinear).to(device)
    if model_summary == True:
        summary(model_unet, (channels, size_boxes, size_boxes))
    
    optimizer = torch.optim.Adam(model_unet.parameters(), lr=lr)
    #Ajust learing rate
    #Decays the learning rate of each parameter group by gamma every step_size epochs.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    min_loss = torch.tensor(float('inf'))
    
    save_losses = []
    scheduler_counter = 0
    
    for epoch in range(N_EPOCHS):
        # training
        model_unet.train()
        loss_list = []
        acc_list = []
        for batch_i, (x, y) in enumerate(train_dataloader):
        
            pred_mask = model_unet(x.to(device))  
            loss = criterion(pred_mask, y.to(device))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            acc_list.append(utils.acc(y,pred_mask).numpy())
    
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [Loss: %f (%f)]"
                % (
                    epoch,
                    N_EPOCHS,
                    batch_i,
                    len(train_dataloader),
                    loss.cpu().detach().numpy(),
                    np.mean(loss_list),
                )
            )
        scheduler_counter += 1
    
        # testing
        model_unet.eval()
        val_loss_list = []
        val_acc_list = []
        val_overall_pa_list = []
        val_per_class_pa_list = []
        val_jaccard_index_list = []
        val_dice_index_list = []
        for batch_i, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():    
                pred_mask = model_unet(x.to(device))  
            val_loss = criterion(pred_mask, y.to(device))
            pred_mask_class = torch.argmax(pred_mask, axis=1)
    
            val_overall_pa, val_per_class_pa, val_jaccard_index, val_dice_index = utils.eval_metrics_sem(y.to(device), pred_mask_class.to(device), 5, device)
            val_overall_pa_list.append(val_overall_pa.cpu().detach().numpy())
            val_per_class_pa_list.append(val_per_class_pa.cpu().detach().numpy())
            val_jaccard_index_list.append(val_jaccard_index.cpu().detach().numpy())
            val_dice_index_list.append(val_dice_index.cpu().detach().numpy())
            val_loss_list.append(val_loss.cpu().detach().numpy())
            val_acc_list.append(utils.acc(y,pred_mask).numpy())
    
        print(' epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}'.format(epoch, 
                                                                                                        np.mean(loss_list), 
                                                                                                        np.mean(acc_list), 
                                                                                                        np.mean(val_loss_list),
                                                                                                        np.mean(val_acc_list)))
        save_losses.append([epoch, np.mean(loss_list), np.mean(acc_list), np.mean(val_loss_list),  np.mean(val_acc_list),
                            np.mean(val_overall_pa_list), np.mean(val_per_class_pa_list),
                            np.mean(val_jaccard_index_list), np.mean(val_dice_index_list)])
    
        compare_loss = np.mean(val_loss_list)
        is_best = compare_loss < min_loss
        print(min_loss, compare_loss)
        if is_best == True and save_model == True:
            print("Best_model")      
            scheduler_counter = 0
            min_loss = min(compare_loss, min_loss)
            torch.save(model_unet.state_dict(), 'model_params/unet_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))
        
        if scheduler_counter > 5:
            lr_scheduler.step()
            print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
            scheduler_counter = 0
    
    if save_model == True:
        dt = datetime.now()
        dict = OrderedDict()
        dict['root'] = root
        dict['dataset_lenght'] = l
        dict['size_boxes'] = size_boxes
        dict['channels'] = channels
        dict['Number_epochs'] = N_EPOCHS
        dict['Bach_size'] = BACH_SIZE
        dict['Loss'] = loss
        dict['bilinear'] = bilinear
        dict['Optimizer_lr'] = lr
        with open('model_params/Train_params_{}.npy'.format(dt), 'wb') as f:
            np.save(f, dict)
            np.save(f, save_losses)