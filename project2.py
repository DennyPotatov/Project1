import torch
import time
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as f
import matplotlib.pyplot as plt
import matplotlib.cm as cm

device = torch.device("cuda")

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='~/datasets', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='~/datasets', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

image, label = trainset[0]  

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Add skip connection
        out = self.relu(out)
        return out

# ResNet-8 Model
class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet8, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual Blocks
        self.layer1 = ResidualBlock(16, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)

        # Fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# # Fire Module (Basic Block of SqueezeNet)
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)

        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand_activation(torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1))

# SqueezeNet Model for CIFAR-10
class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Initial Conv Layer
        self.relu = nn.ReLU(inplace=True)

        # Fire Modules
        self.fire1 = FireModule(32, 16, 32)
        self.fire2 = FireModule(64, 16, 32)
        self.fire3 = FireModule(64, 32, 64)
        self.fire4 = FireModule(128, 32, 64)

        # Final Layers
        self.conv2 = nn.Conv2d(128, num_classes, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.fire1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)

        x = self.conv2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.softmax(x)
        return x


class MiniVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniVGG, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16x16
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 8x8
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 4x4
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.fc_layers(x)
        return x
    
class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # Define the layers
        self.conv_block = nn.Sequential(
            # First Convolutional Layer
            nn.Conv2d(3, 6, kernel_size=5),   # 32x32 -> 28x28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14

            # Second Convolutional Layer
            nn.Conv2d(6, 16, kernel_size=5),  # 14x14 -> 10x10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 10x10 -> 5x5
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),  # Flatten 5x5x16 -> 120
            nn.Tanh(),
            nn.Linear(120, 84),           # 120 -> 84
            nn.Tanh(),
            nn.Linear(84, num_classes)    # 84 -> 10 (CIFAR-10 classes)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x
    

result = {}
time_result = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = ResNet8().to(device)

for name, model in {"ResNet8": ResNet8(), "SqueezeNet": SqueezeNet(), "LetNet5": LeNet5(), "MiniVGG": MiniVGG()}.items():

    print(f"Start of the training of {name}")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    epochs = 40
    min_loss = float('inf')
    patience = 3  
    no_improvement_count = 0 
    train_loss_list = []
    test_loss_list = []
    train_acc_list = []
    test_acc_list = []

    start_time = time.time() 

    for epoch in range(epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # _, predicted = outputs.max(1)
            _, predicted = torch.max(outputs.data, 1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(trainloader)
        train_acc = correct / total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc * 100:.2f}%")


        if train_loss < min_loss:
            min_loss = train_loss
            no_improvement_count = 0  
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print(f"Training stopped early at epoch {epoch+1} due to minimal improvement.")
            break

        correct = 0.0
        total = 0.0
        test_loss = 0.0
        model.eval() 
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += labels.eq(predicted).sum().item()
                total += labels.size(0)


        test_loss /= len(testloader)
        test_acc = correct / total
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        print(f"Test Loss: {test_loss} Test Accuracy: {test_acc * 100}%")

    
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} training and testing time: {epoch_time:.2f} sec")

    # Total training time for the model
    total_training_time = time.time() - start_time

    print(f"Total Training Time for {name}: {total_training_time:.2f} sec")

    model.eval()
    with torch.no_grad():
        images = None
        for images_batch, labels in testloader:
            images = images_batch.to(device) 
            break 

        torch.cuda.synchronize()  
        start_time = time.time()
        output = model(images)  
        torch.cuda.synchronize()  
        end_time = time.time()

        inference_time = end_time - start_time
        print(f"Inference time for {name}: {inference_time:.5f} sec")

    time_result[name] = (total_training_time, inference_time)
    result[name] = (train_loss_list, test_loss_list, train_acc_list, test_acc_list)


colors = cm.get_cmap("tab10", len(result)) 

plt.figure(figsize=(12, 5)) 

for i, (name, (train_loss, test_loss, train_acc, test_acc)) in enumerate(result.items()):
    print(train_loss, test_loss)
    color = colors(i) 
    plt.plot(train_loss, color=color,  label=f"{name} Train Loss")
    plt.plot(test_loss, color=color,  linestyle="dashed", label=f"{name} Test Loss")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve Comparison")
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))  
plt.grid()

plt.savefig("loss_curve.png", bbox_inches="tight") 

plt.figure(figsize=(12, 5)) 

plt.figure(figsize=(12, 5)) 

for i, (name, (train_loss, test_loss, train_acc, test_acc)) in enumerate(result.items()):
    print(train_acc, test_acc)
    color = colors(i) 
    plt.plot(train_acc, color=color, label=f"{name} Train Accuracy")
    plt.plot(test_acc, color=color, linestyle="dashed", label=f"{name} Test Accuracy")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve Comparison")
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1))  
plt.grid()

plt.savefig("accuracy_curve.png", bbox_inches="tight") 

plt.figure(figsize=(12, 5)) 

plt.bar(list(time_result.keys()), [item[0] for item in time_result.values()] , width=0.4, label="Training Time", align='center')

plt.xlabel("Model")
plt.ylabel("Time (seconds)")
plt.title("Training Time Comparison")
plt.legend()
plt.grid(True)

plt.savefig("training_time.png")


plt.figure(figsize=(12, 5)) 

plt.bar(list(time_result.keys()), [item[1] for item in time_result.values()], width=0.4, label="Inference Time", align='center')

plt.xlabel("Model")
plt.ylabel("Inference Time (seconds)")
plt.title("Inference Time Comparison")
plt.legend()
plt.grid(True)

plt.savefig("inference_time.png")
