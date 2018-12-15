import csv
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

# Data
transform = transforms.Compose(
    [transforms.Resize([224, 224]),
     transforms.ToTensor()])

train_dataset = datasets.CocoDetection(
    root='/mnt/hdd-4tb/abhay/cocostuff/dataset/images/train2017',
    annFile='/mnt/hdd-4tb/abhay/cocostuff/dataset/annotations/stuff_train2017.json',
    transform=transform)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=50, shuffle=True)

# Model
model = models.resnet152(pretrained=False)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

num_epochs = 1
learning_rate = 0.001

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    step = 0
    data_times = []
    compute_times = []
    start = time.time()
    for images, labels in train_loader:
        data_times.append(time.time() - start)
        start = time.time()
        step += 1
        if step == total_step:
            break

        images = images.to(device)
        labels = torch.ones([50, 1000])
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        compute_times.append(time.time() - start)
        print("Step = " + str(step) + " out of " + str(total_step))
        start = time.time()

    with open('logs.csv', 'w', newline='') as logfile:
        logwriter = csv.writer(
            logfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in zip(data_times, compute_times):
            logwriter.writerow(row)
