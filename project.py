import torchvision
import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from PIL import Image



device = torch.device("cuda:0")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize((224, 224)),  
    transforms.ToTensor(), 
])

# Step 1 Download dataset ###################################################################################
dataset = torchvision.datasets.Caltech101(
    root="~/datasets",  
    download=True,
    transform=transform
)


print(dataset.categories)  

# Step 2 Choose tree categories ###################################################################################
selected_categories = {"butterfly", "flamingo", "dolphin"}

# Step 3 Load dataset ###################################################################################
filtered_data = [(img, label) for img, label in dataset if dataset.categories[label] in selected_categories]

# Step 4 Count number of images per category ###################################################################################
category_counts = Counter([dataset.categories[label] for _, label in filtered_data])
print(category_counts)

# Step 5 Determin the smallers amount of images per category ###################################################################################
minSetCount = min(category_counts.values()) 
print("Minimum number of images per category:", minSetCount)

# Step 6 Adjust each category to have the same number of images ###################################################################################

balanced_dataset = []
for category in selected_categories:
    category_images = [(img, label) for img, label in filtered_data if dataset.categories[label] == category]
    balanced_dataset.extend(random.sample(category_images, minSetCount))

# Step 7 Download the pretrained network ###################################################################################
net = models.resnet50(pretrained=True)
print(net)  
print(net.conv1) 
w1 = net.conv1.weight  
print(w1.shape) 

# Step 8 Extract the training features in CNN ###################################################################################

w1 = net.conv1.weight.detach().cpu().numpy()

num_filters = w1.shape[0]

kernel_size = w1.shape[2]

rows = int(np.sqrt(num_filters))
cols = int(np.ceil(num_filters / rows))

fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

for i in range(num_filters):
    r = i // cols
    c = i % cols
    
    filter_img = w1[i, :, :, :].transpose(1, 2, 0)
    filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min())

    if rows == 1 and cols == 1:
        axes.imshow(filter_img)
    elif rows == 1 or cols == 1:
        axes[max(r, c)].imshow(filter_img)
    else:
        axes[r, c].imshow(filter_img)
    axes[r, c].axis('off')

plt.tight_layout()
plt.savefig("resnet50_conv1_filters.png") 


# Step 9 Evaluate Classification (Wasn't specified what classifier to use)###################################################################################

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


resnet = models.resnet50(pretrained=True).to(device)
resnet.eval() 


def extract_features(dataset, model, device):
    features = []
    with torch.no_grad():  
        for image, _ in dataset:
            image = image.unsqueeze(0).to(device)  
            output = model.conv1(image) 
            output = model.bn1(output)
            output = model.relu(output)
            output = model.maxpool(output)
            
            output = model.layer1(output)
            output = model.layer2(output)
            output = model.layer3(output)
            output = model.layer4(output)
            
            output = model.avgpool(output) 
            output = torch.flatten(output, 1) 
            features.append(output.cpu().numpy()) 
    return np.concatenate(features, axis=0) 


train_features = extract_features(train_dataset, resnet, device)
test_features = extract_features(test_dataset, resnet, device)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for i in range(train_features.shape[0]):
    ax1.plot(train_features[i, :])

ax1.set_title("Train Features")
ax1.set_xlabel("Feature Index")
ax1.set_ylabel("Feature Value")

for i in range(test_features.shape[0]):
    ax2.plot(test_features[i, :])

ax2.set_title("Test Features")
ax2.set_xlabel("Feature Index")
ax2.set_ylabel("Feature Value")

plt.tight_layout()

plt.savefig("train_test_features_gpu.png")

dataloader = DataLoader(balanced_dataset, batch_size=32, shuffle=True)

feature_extractor = nn.Sequential(*list(net.children())[:-1])
feature_extractor.eval()  

features = []
labels = []

with torch.no_grad():
    for images, label in dataloader:
        output = feature_extractor(images) 
        output = output.view(output.size(0), -1) 
        features.append(output)
        labels.append(label)

features = torch.cat(features)
labels = torch.cat(labels)

print("Extracted Feature Shape:", features.shape)  

X = features.numpy()
y = labels.numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Classification Accuracy:", acc)

confMat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confMat)

per_class_accuracy = np.diag(confMat) / np.sum(confMat, axis=1)
mean_accuracy = np.mean(per_class_accuracy)

print("Mean Accuracy:", mean_accuracy)

# Step 10 Apply the trained classifier to one test image ###################################################################################

img_path = "image_0011.jpg"
image = Image.open(img_path)

transformed_image = transform(image).unsqueeze(0) 

feature_extractor.eval() 

with torch.no_grad():
    extracted_feature = feature_extractor(transformed_image)
    extracted_feature = extracted_feature.view(1, -1)


feature_np = extracted_feature.numpy()

predicted_label = clf.predict(feature_np)[0] 

predicted_class = dataset.categories[predicted_label]

print("Predicted Class:", predicted_class)


