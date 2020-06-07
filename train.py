import torch
import torch.nn as nn
import torch.optim as optim
from datasets.face300w import face300W
from models.hourglass import hg
from losses.loss import Loss_weighted
import cv2
import numpy as np

#basic setting
num_epochs = 240
vis_result = True
batch_size = 8
pretrained = True
pretrained_start_epoch = 110
start_lr = 1e-6
pretrain_path = "./ckpt/100.pth"
if(vis_result):
    import matplotlib.pyplot as plt
W = 5
omega = 300
epsilon= 2

num_epochs -= pretrained_start_epoch

#model load
if pretrained:
    model = torch.load(pretrain_path)
else:
    model  = hg(num_stacks=4,
                num_blocks=2,
                num_classes=68+1)
model = model.cuda()
print("model loaded")

#dataset load
dataset_path = "./data/300W"
train_dataset = face300W(dataset_path)
dataiter = len(train_dataset)//batch_size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
print("images : ",len(train_dataset))

#loss define
criterion = Loss_weighted()

#optim setting
optimizer = optim.RMSprop(model.parameters(), lr=start_lr, weight_decay=1e-5, momentum=0)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[60,120], gamma=0.1)

#train start!
print("train start!")

for epoch in range(num_epochs):
    losses = list()
    for iteration,sample in enumerate(train_loader):
        img,hmap,M,pts = sample
        img = img.permute(0,3,1,2)
        img,hmap,M = img.cuda(), hmap.cuda(), M.cuda()

        out = model(img)

        loss = sum(criterion(o, hmap, M) for o in out)
        losses.append(loss.item())
        print(str(epoch)," :: ",str(iteration), "/",dataiter,"\n  loss     :: ",loss.item())
        print("  avg loss :: ",sum(losses)/len(losses))

        # Backward pass and parameter update.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if vis_result and (iteration %20 == 0):
            y_hat = out[-1][0,:68].detach().cpu().numpy()
            plt.imshow(img.permute(0,2,3,1)[0].cpu().numpy())
            plt.imshow(cv2.resize(np.max(y_hat, axis=0), dsize=(256, 256), interpolation=cv2.INTER_LINEAR),alpha=0.5)
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    if(epoch %10 == 0):
        torch.save(model, "./ckpt/"+str(epoch)+".pth")

    scheduler.step()
