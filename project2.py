import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as f

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

# print(image.shape)  

class CNN1(nn.Module):
    def __init__(self, num_classes):
        super (CNN1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1)

        self.avgp = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fullcl1 = nn.Linear(512*4*4, 256)
        self.fullcl2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.avgp(f.relu(self.conv1(x)))
        x = self.avgp(f.relu(self.conv2(x)))
        x = self.avgp(f.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = f.relu(self.fullcl1(x))
        x = self.fullcl2(x)

        return x


# class CNN2(nn.Module):
#     def __init__(self, num_classes):
#         super(CNN2, self).__init__()

#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # C11
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # C12
#         self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # First C13
#         self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # Second C13 (for next iteration)
#         self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # Third C13

#         # Average Pooling layer
#         self.avgp = nn.AvgPool2d(kernel_size=2, stride=2)  # Pa

#         # Fully connected layers
#         self.fullcl1 = nn.Linear(256 * 1 * 1, 128)  # Adjust input size to match feature map size
#         self.fullcl2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         # Apply conv1, conv2, and conv3_1 (first conv3) followed by average pooling
#         x = self.avgp(f.relu(self.conv1(x)))  # conv1 → avgpool
#         x = self.avgp(f.relu(self.conv2(x)))  # conv2 → avgpool
        
#         # Apply first conv3 (256 channels output)
#         x = self.avgp(f.relu(self.conv3_1(x)))  # First conv3 → avgpool

#         # Apply second conv3 (128 channels input for next iteration)
#         x = self.avgp(f.relu(self.conv3_2(x)))  # Second conv3 → avgpool

#         # Apply third conv3 (128 channels input)
#         x = self.avgp(f.relu(self.conv3_3(x)))  # Third conv3 → avgpool

#         # Flatten the tensor before passing it to fully connected layers
#         x = x.view(x.size(0), -1)  # Flatten the output
        
#         # Fully connected layers
#         x = f.relu(self.fullcl1(x))  # First fully connected layer
#         x = self.fullcl2(x)  # Output layer
        
#         return x


# class CNN2(nn.Module):
#     def __init__(self, num_classes):
#         super(CNN2, self).__init__()

#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # C11
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # C12
#         self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # C13 (this should output 256 channels)

#         # Average Pooling layer
#         self.avgp = nn.AvgPool2d(kernel_size=2, stride=2)  # Pa

#         # Fully connected layers - Adjust input size based on the feature map output
#         self.fullcl1 = nn.Linear(256 * 4 * 4, 128)  # Adjusted input size
#         self.fullcl2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         # Apply conv1 and conv2 followed by average pooling
#         x = self.avgp(f.relu(self.conv1(x)))  # conv1 → avgpool
#         x = self.avgp(f.relu(self.conv2(x)))  # conv2 → avgpool
        
#         # Apply conv3 repeatedly while keeping output channels 256 (it should always output 256)
#         # Note: Input to conv3 should always be 128 channels (after conv2), and it will output 256 channels.
#         x = f.relu(self.conv3(x))  # First conv3 → no avgpool here
#         x = self.avgp(x)  # Apply pooling after conv3

#         x = f.relu(self.conv3(x))  # Second conv3 → no avgpool here
#         x = self.avgp(x)  # Apply pooling again

#         x = f.relu(self.conv3(x))  # Third conv3 → no avgpool here
#         x = self.avgp(x)  # Apply pooling again

#         # Flatten the tensor before passing it to fully connected layers
#         x = x.view(x.size(0), -1)  # Flatten the output
        
#         # Fully connected layers
#         x = f.relu(self.fullcl1(x))  # First fully connected layer
#         x = self.fullcl2(x)  # Output layer
        
#         return x


class CNN3(nn.Module):
    def __init__(self, num_classes):
        super(CNN3, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)  # C11
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # C12
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)  # First C13
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)  # Second C13 (for next iteration)

        # Average Pooling layer
        self.avgp = nn.AvgPool2d(kernel_size=2, stride=2)  # Pa

        # Fully connected layers
        self.fullcl1 = nn.Linear(256 * 2 * 2 , 128)  # Adjust input size to match feature map size
        self.fullcl2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Apply conv1, conv2, and conv3_1 (first conv3) followed by average pooling
        x = self.avgp(f.relu(self.conv1(x)))  # conv1 → avgpool
        x = self.avgp(f.relu(self.conv2(x)))  # conv2 → avgpool
        
        # Apply first conv3 (256 channels output)
        x = self.avgp(f.relu(self.conv3_1(x)))  # First conv3 → avgpool

        # Apply second conv3 (128 channels input for next iteration)
        x = self.avgp(f.relu(self.conv3_2(x)))  # Second conv3 → avgpool

        # Flatten the tensor before passing it to fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the output
        
        # Fully connected layers
        x = f.relu(self.fullcl1(x))  # First fully connected layer
        x = self.fullcl2(x)  # Output layer
        
        return x
    
# print(f"Using device: {device}")

class CNN4(nn.Module):
    def __init__(self, num_classes):
        super(CNN4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=1024, kernel_size=3, padding=1)

        self.avgp = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fullcl1 = nn.Linear(512*2*2, 128)  # Flattened size = 1024*2*2 = 4096
        self.fullcl2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.avgp(f.relu(self.conv1(x)))  # After conv1 
        x = self.avgp(f.relu(self.conv2(x)))  # After conv2
        x = self.avgp(f.relu(self.conv4(x)))  # After conv3
        x = self.avgp(f.relu(self.conv3(x)))  # After conv4

        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 1024*2*2)
        
        # Fully connected layers
        x = f.relu(self.fullcl1(x))  # First fully connected layer
        x = self.fullcl2(x)          # Output layer

        return x




# model = CNN4(10).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# epochs = 40
# min_loss = float('inf')  # Initialize with a large number to track the minimum loss
# patience = 3  # Number of epochs to wait for improvement before stopping
# no_improvement_count = 0  # Counter for epochs without improvement

# for epoch in range(epochs):
#     running_loss = 0.0
#     for inputs, labels in trainloader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     avg_loss = running_loss / len(trainloader)
#     print(f"Epoch {epoch+1}, Loss: {avg_loss}")

#     # Check if the loss has improved
#     if avg_loss < min_loss:
#         min_loss = avg_loss
#         no_improvement_count = 0  # Reset counter
#     else:
#         no_improvement_count += 1

#     # If there's no significant improvement for 'patience' epochs, stop training
#     if no_improvement_count >= patience:
#         print(f"Training stopped early at epoch {epoch+1} due to minimal improvement.")
#         break

# correct = 0
# total = 0
# model.eval()  # Set model to evaluation mode
# with torch.no_grad():
#     for images, labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         # Skip connection
#         self.skip = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.skip = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         identity = self.skip(x)
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += identity  # Add skip connection
#         out = self.relu(out)
#         return out

# # ResNet-8 Model
# class ResNet8(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet8, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.relu = nn.ReLU(inplace=True)

#         # Residual Blocks
#         self.layer1 = ResidualBlock(16, 32, stride=1)
#         self.layer2 = ResidualBlock(32, 64, stride=2)
#         self.layer3 = ResidualBlock(64, 128, stride=2)

#         # Fully connected layer
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)

#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNet8().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training Loop
# num_epochs = 20
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct, total = 0, 0

#     for images, labels in trainloader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# # Testing Loop
# model.eval()
# correct, total = 0, 0

# with torch.no_grad():
#     for images, labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms

# # Fire Module (Basic Block of SqueezeNet)
# class FireModule(nn.Module):
#     def __init__(self, in_channels, squeeze_channels, expand_channels):
#         super(FireModule, self).__init__()
#         self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
#         self.squeeze_activation = nn.ReLU(inplace=True)

#         self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
#         self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)

#         self.expand_activation = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.squeeze_activation(self.squeeze(x))
#         return self.expand_activation(torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1))

# # SqueezeNet Model for CIFAR-10
# class SqueezeNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SqueezeNet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Initial Conv Layer
#         self.relu = nn.ReLU(inplace=True)

#         # Fire Modules
#         self.fire1 = FireModule(32, 16, 32)
#         self.fire2 = FireModule(64, 16, 32)
#         self.fire3 = FireModule(64, 32, 64)
#         self.fire4 = FireModule(128, 32, 64)

#         # Final Layers
#         self.conv2 = nn.Conv2d(128, num_classes, kernel_size=1)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.relu(self.conv1(x))

#         x = self.fire1(x)
#         x = self.fire2(x)
#         x = self.fire3(x)
#         x = self.fire4(x)

#         x = self.conv2(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.softmax(x)
#         return x

# # Check model structure
# # model = SqueezeNet()
# # print(model)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SqueezeNet().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training Loop
# num_epochs = 25
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct, total = 0, 0

#     for images, labels in trainloader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# # Testing Loop
# model.eval()
# correct, total = 0, 0

# with torch.no_grad():
#     for images, labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")


# class MiniVGG(nn.Module):
#     def __init__(self, num_classes=10):
#         super(MiniVGG, self).__init__()

#         self.conv_block1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 16x16
#         )

#         self.conv_block2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 8x8
#         )

#         self.conv_block3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 4x4
#         )

#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 4 * 4, 256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.conv_block1(x)
#         x = self.conv_block2(x)
#         x = self.conv_block3(x)
#         x = self.fc_layers(x)
#         return x
    

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = MiniVGG().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training Loop
# num_epochs = 20
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct, total = 0, 0

#     for images, labels in trainloader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# # Testing Loop
# model.eval()
# correct, total = 0, 0

# with torch.no_grad():
#     for images, labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")



# class LeNet5(nn.Module):
#     def __init__(self, num_classes=10):
#         super(LeNet5, self).__init__()

#         # Define the layers
#         self.conv_block = nn.Sequential(
#             # First Convolutional Layer
#             nn.Conv2d(3, 6, kernel_size=5),   # 32x32 -> 28x28
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14

#             # Second Convolutional Layer
#             nn.Conv2d(6, 16, kernel_size=5),  # 14x14 -> 10x10
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2, stride=2),  # 10x10 -> 5x5
#         )

#         self.fc_block = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(16 * 5 * 5, 120),  # Flatten 5x5x16 -> 120
#             nn.Tanh(),
#             nn.Linear(120, 84),           # 120 -> 84
#             nn.Tanh(),
#             nn.Linear(84, num_classes)    # 84 -> 10 (CIFAR-10 classes)
#         )

#     def forward(self, x):
#         x = self.conv_block(x)
#         x = self.fc_block(x)
#         return x
    

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = LeNet5().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training Loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     correct, total = 0, 0

#     for images, labels in trainloader:
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {100 * correct/total:.2f}%")

# # Testing Loop
# model.eval()
# correct, total = 0, 0

# with torch.no_grad():
#     for images, labels in testloader:
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")