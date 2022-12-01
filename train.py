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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run(root, l, size_boxes, channels, N_EPOCHS, BACH_SIZE, loss_str, scale=1, lr = 1e-3, 
        save_model=False, bilinear=False, model_summary=False, dropout=False):

    CE_weights = torch.Tensor([1.0,10.0,10.0,10.0,1.0]).to(device)

    if loss_str == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(weight=CE_weights).to(device)
    if loss_str == 'FocalLoss':
        criterion = losses.FocalLoss(gamma=10, alpha=CE_weights).to(device)
    if loss_str == 'mIoU':
        criterion = losses.mIoULoss(n_classes=5, weight=CE_weights).to(device)

    test_num = int(0.05 * l)
    print("Training set")
    data_train=dataset.segDataset(root+'Train/', l=l-test_num, s=size_boxes)
    print("Validating set")
    data_test=dataset.segDataset_val(root+'Validate/', l=test_num, s=size_boxes)
    
    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=BACH_SIZE, shuffle=True, num_workers=1)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=BACH_SIZE, shuffle=False, num_workers=1)
    
    n_class = len(data_train.bin_classes)
    
    model_unet = model.UNet(n_channels=channels, n_classes=n_class, scale=scale, bilinear=bilinear, dropout=dropout).to(device)
    if model_summary == True:
        summary(model_unet, (channels, size_boxes, size_boxes))
    
    optimizer = torch.optim.Adam(model_unet.parameters(), lr=lr)
    #Ajust learing rate
    #Decays the learning rate of each parameter group by gamma every step_size epochs.
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    min_loss = torch.tensor(float('inf'))
    
    save_losses = []
    #Histograms
    save_h_train_losses = []
    save_h_val_losses = []
    scheduler_counter = 0
    
    for epoch in range(N_EPOCHS):
        # training
        model_unet.train()
        loss_list = []
        acc_list = []

        train_xypred = []

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

            #Evaluation of the training results
            pred_mask_class = torch.argmax(pred_mask, axis=1)
            if batch_i == len(train_dataloader) - 1:
                x_p = x.cpu().detach().numpy()
                y_p = y.cpu().detach().numpy()
                pred_p = pred_mask_class.cpu().detach().numpy()
                train_xypred.append([x_p[-1,0,:,:,:], y_p[-1], pred_p[-1]])

        scheduler_counter += 1
    
        # testing
        model_unet.eval()
        val_loss_list = []
        val_acc_list = []
        val_overall_pa_list = []
        val_per_class_pa_list = []
        val_jaccard_index_list = []
        val_dice_index_list = []

        test_xypred = []

        for batch_i, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():    
                pred_mask = model_unet(x.to(device))  
            val_loss = criterion(pred_mask, y.to(device))
            pred_mask_class = torch.argmax(pred_mask, axis=1)

            #Evaluation of the testing results
            if batch_i == len(test_dataloader) - 1:
                x_p = x.cpu().detach().numpy()
                y_p = y.cpu().detach().numpy()
                pred_p = pred_mask_class.cpu().detach().numpy()
                test_xypred.append([x_p[-1,0,:,:,:], y_p[-1], pred_p[-1]])

    
            val_overall_pa, val_per_class_pa, val_jaccard_index, val_dice_index, pc_opa, pc_j, pc_d= utils.eval_metrics_sem(y.to(device), pred_mask_class.to(device), 5, device)
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
        if epoch % 10 == 0:
            print("Partial Model")

            save_h_train_losses.append([loss_list, acc_list])
            save_h_val_losses.append([val_loss_list, val_acc_list])

            torch.save(model_unet.state_dict(), 'model_params/unet_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))

            x_p = train_xypred[0][0]
            y_p = train_xypred[0][1]
            pred_p = train_xypred[0][2]

            values = [0,1,2,3,4]
            bin_classes = data_train.bin_classes

            i_cmap=plt.get_cmap('PiYG', 5)
            list_cmap = i_cmap(range(5))

            fig, ax = plt.subplots(nrows=1, ncols=7, sharex=True, sharey=True)
            fig.set_size_inches(15, 5)
            for i in range(5):
                im=ax[i].imshow(x_p[i,:,:], origin='lower', cmap='gray')
            l_values1 = np.unique(y_p)
            l_values2 = np.unique(pred_p)
            im1=ax[-2].imshow(y_p, origin='lower', cmap=ListedColormap(list_cmap[l_values1]))
            im2=ax[-1].imshow(pred_p, origin='lower', cmap=ListedColormap(list_cmap[l_values2]))
            ax[-2].set_title('{}'.format(l_values1))
            ax[-1].set_title('{}'.format(l_values2))

            colors = [list_cmap[value] for value in values]
            patches = [mpatches.Patch(color=colors[i], 
            label="{l}".format(l=bin_classes[i])) for i in range(len(bin_classes))]
            lgd = plt.legend(handles=patches, bbox_to_anchor=(2.5, 0.75), loc=1, borderaxespad=0. , ncol=1)

            fig.suptitle('Model_Train_Epoch_{}'.format(epoch))
            plt.tight_layout()
            plt.savefig("Train_epoch_{}.png".format(epoch), dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')

            x_p = test_xypred[0][0]
            y_p = test_xypred[0][1]
            pred_p = test_xypred[0][2]

            fig, ax = plt.subplots(nrows=1, ncols=7, sharex=True, sharey=True)
            fig.set_size_inches(15, 5)
            for i in range(5):
                im=ax[i].imshow(x_p[i,:,:], origin='lower', cmap='gray')
            l_values1 = np.unique(y_p)
            l_values2 = np.unique(pred_p)
            im1=ax[-2].imshow(y_p, origin='lower', cmap = ListedColormap(list_cmap[l_values1]))
            im2=ax[-1].imshow(pred_p, origin='lower', cmap = ListedColormap(list_cmap[l_values2]))
            ax[-2].set_title('{}'.format(l_values1))
            ax[-1].set_title('{}'.format(l_values2))

            colors = [list_cmap[value] for value in values]
            patches = [mpatches.Patch(color=colors[i], 
            label="{l}".format(l=bin_classes[i])) for i in range(len(bin_classes))]
            lgd = plt.legend(handles=patches, bbox_to_anchor=(2.5, 0.75), loc=1, borderaxespad=0. , ncol=1)

            fig.suptitle('Model_test_Epoch_{}'.format(epoch))
            plt.tight_layout()
            plt.savefig("Test_epoch_{}.png".format(epoch), dpi=200, bbox_extra_artists=(lgd,), bbox_inches='tight')


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
        
        if epoch == 199:
            print("Final Model")
            torch.save(model_unet.state_dict(), 'model_params/unet_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))
    
    if save_model == True:
        dt = datetime.now()
        dict = OrderedDict()
        dict['root'] = root
        dict['dataset_lenght'] = l
        dict['size_boxes'] = size_boxes
        dict['channels'] = channels
        dict['Number_epochs'] = N_EPOCHS
        dict['Bach_size'] = BACH_SIZE
        dict['Loss'] = loss_str
        dict['bilinear'] = bilinear
        dict['Optimizer_lr'] = lr
        dict['Scale'] = scale
        with open('model_params/Train_params_{}.npy'.format(dt), 'wb') as f:
            np.save(f, dict)
            np.save(f, save_losses)
            np.save(f, save_h_train_losses)
            np.save(f, save_h_val_losses)