import dataset
import losses
import model
import utils
import numpy as np
import sys
import torch
from torch import nn
from datetime import datetime

device = 'cuda' if torch.cuda.is_available() else 'cpu'

root = 'data/Masks_S_v2/'
N_EPOCHS = 30
BACH_SIZE = 16
loss = 'FocalLoss'
save_model = True
bilinear = False
lr = 1e-3

if loss == 'CrossEntropy':
    criterion = nn.CrossEntropyLoss().to(device)
if loss == 'FocalLoss':
    criterion = losses.FocalLoss(gamma=3/4).to(device)
if loss == 'mIoU':
    criterion = losses.mIoULoss(n_classes=5).to(device)

data=dataset.segDataset(root)

print('Number of data : '+ str(len(data)))

test_num = int(0.1 * len(data))

train_dataset, test_dataset = torch.utils.data.random_split(data, [len(data)-test_num, test_num], generator=torch.Generator().manual_seed(101))
N_DATA, N_TEST = len(train_dataset), len(test_dataset)
print(f'Data : {N_DATA}')
print(f'Test : {N_TEST}')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BACH_SIZE, shuffle=True, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BACH_SIZE, shuffle=False, num_workers=1)

model = model.UNet(n_channels=1, n_classes=5, bilinear=False).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#Ajust learing rate
#Decays the learning rate of each parameter group by gamma every step_size epochs.
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
min_loss = torch.tensor(float('inf'))

save_losses = []
scheduler_counter = 0

for epoch in range(N_EPOCHS):
    # training
    model.train()
    loss_list = []
    acc_list = []
    for batch_i, (x, y) in enumerate(train_dataloader):

        pred_mask = model(x.to(device))  
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
    model.eval()
    val_loss_list = []
    val_acc_list = []
    val_overall_pa_list = []
    val_per_class_pa_list = []
    val_jaccard_index_list = []
    val_dice_index_list = []
    for batch_i, (x, y) in enumerate(test_dataloader):
        with torch.no_grad():    
            pred_mask = model(x.to(device))  
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
        torch.save(model.state_dict(), 'model_params/unet_epoch_{}_{:.5f}.pt'.format(epoch,np.mean(val_loss_list)))
    
    if scheduler_counter > 5:
        lr_scheduler.step()
        print(f"lowering learning rate to {optimizer.param_groups[0]['lr']}")
        scheduler_counter = 0
        
if save_model == True:
    dt = datetime.now()
    np.save('model_params/Train_params_{}.npy'.format(dt), save_losses)