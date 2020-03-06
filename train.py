import torch
import torch.nn as nn
import torch.optim as optim
from datasets.face300w import face300W
from models.hourglass import hg
from losses.loss import Loss_weighted

#basic setting
num_epochs = 240

#model load
model  = hg(num_stacks=4,
            num_blocks=2,
            num_classes=68+1)

#dataset load
dataset_path = "./data/300w"
train_dataset = face300W(dataset_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                           shuffle=True, num_workers=0)

#loss define
criterion = Loss_weighted()

#optim setting
optimizer = optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-5, momentum=0)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[80,160], gamma=0.1)

#train start!
print("train start!")

for epoch in range(num_epochs):
    for iteration,sample in enumerate(train_loader):
        img,hmap,M,pts = sample
        img = img.transpose(1,-1)

        out = model(img)

        import matplotlib.pyplot as plt
        lossmap = out[0][0,-1].detach().numpy()
        plt.imshow(lossmap)
        plt.show()

        loss = sum(criterion(o, hmap, M) for o in out)
        print(loss.item())

        # Backward pass and parameter update.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
