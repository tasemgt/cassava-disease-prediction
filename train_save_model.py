import os
from PIL import Image
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from random import shuffle

import torch

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T

from torchvision.models import resnet18, vgg16, vgg13
from sklearn.metrics import accuracy_score, roc_auc_score



data_path = "data/train"

#Exploring Dataset

# Dataset classes
print(os.listdir(data_path))

# Class paths
data_paths = [(p, os.path.join(data_path, p)) for p in os.listdir(data_path)]
data_paths = [p for p in data_paths if p[0] != '.DS_Store'] #Remove .DS_Store since it isnt a folder


sample_files = os.listdir(data_paths[0][-1])

sample_paths = [
    os.path.join(data_paths[0][-1], f)
    for f in sample_files
]

image_sizes = []

for f in sample_paths:
    img = np.array(Image.open(f)).astype(int)
    image_sizes.append(img.shape) #img.size also


set(image_sizes)

sample_files = [os.listdir(p[-1])[:4] for p in data_paths if os.path.isdir(p[-1])]


# In[43]:


sample_files


# In[44]:


data_paths


# In[45]:


sample_paths = list(map(lambda x, y: {x[0]: [os.path.join(x[1], y_) for y_ in y]}, data_paths, sample_files))



print("Is GPU Available:", torch.cuda.is_available())


# Set computational device as either CPU or GPU (i.e., CUDA)
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Using device:", DEVICE)


# Dataset Preparation
class CassavaDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose(
                [
                    T.Resize((224, 224)), # Resize images
                    T.ToTensor(), # Convert images to PyTorch tensors
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize images (Values used for imagenet)
                ]
            )
        classes = os.listdir(path)
        classes = [c for c in classes if c != '.DS_Store'] #Remove .DS_Store since it isnt a folder
        print(classes)
        self.class_map = dict(zip(classes, [_ for _ in range(len(classes))]))
        self.files = []

        for class_ in classes:
            self.files +=[(os.path.join(path, class_, f), class_) for f in os.listdir(os.path.join(path, class_))]

        shuffle(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, ix):
        file = self.files[ix]
        image = Image.open(file[0])

        return self.transform(image), self.class_map[file[-1]]


# Instantiae Dataset object
dataset = CassavaDataset(path = "data/train")

# Number of samples in Dataset
num_samples = len(dataset)


print(num_samples)


# Set test size
test_size = int(num_samples * .30)

# Split dataset into train and test splits
train_ds, test_ds = random_split(dataset = dataset, lengths = [.7, .3])

len(train_ds)

len(test_ds)

BATCH_SIZE = 16

# Generate DataLoaders for faster training in batches...
train_dl, test_dl = (
    DataLoader(dataset = train_ds, batch_size = BATCH_SIZE, shuffle = True),
    DataLoader(dataset = test_ds, batch_size = BATCH_SIZE, shuffle = True)
)


def generate_model(out_features, freeze_weights = True, model_function = resnet18):
    base_model = model_function(weights = True)

    if freeze_weights:
        for param in base_model.parameters():
            param.requires_grad_(False)

    # Try to tweak base model to what we want (out_features)
    try:
        # For resnet model
        in_features = base_model.fc.in_features
        new_layer = nn.Linear(in_features, out_features)
        base_model.fc = new_layer
    except:
        # For vgg-13 model
        in_features = base_model.classifier[0].in_features
        new_layer = nn.Linear(in_features, out_features)
        base_model.classifier = new_layer
    
    return base_model



class Model(nn.Module):
    def __init__(self, out_features, freeze_weights = True, model_function=resnet18):
        super().__init__()

        self.base = generate_model(out_features = out_features, freeze_weights = freeze_weights, model_function = model_function)
    
    def forward(self, x):
        x = self.base(x)
        return torch.softmax(x, dim = -1)


# Training hyperparameters
EPOCHS = 20
LR = 1e-3
criterion = nn.CrossEntropyLoss()

NUM_CLASSES = len(dataset.class_map) # Number of data catagories

# Optimizer hyperparameters
FACTOR = 10
AMSGRAD = False
BETAS = (.9, .999)



def initialize_model_weights(model, init_func = nn.init.normal_):
    for name, params in model.named_parameters():
        if name in ["fc", "classifier"]:
            init_func(params)
        else:
            continue
    
    return model


def training_loop(epochs, model, optimizer):
    TRAIN_LOSSES, TEST_LOSSES = [], []
    TRAIN_ACCS, TEST_ACCS = [], []
    
    for epoch in range(epochs):
        train_losses, test_losses = [], []
        train_accs, test_accs = [], []
        
        model.train() # Set model in training mode
        
        for X, y in iter(train_dl):
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            y_pred = model(X.to(DEVICE))
            train_loss = criterion(y_pred, y) #Compare actual targets and predicted targets to get the loss
            train_loss.backward() #Back ppropagate the loss
            optimizer.step()
            optimizer.zero_grad()
            
            train_losses.append(train_loss.item())
    
            train_acc = accuracy_score(y.cpu().numpy(), y_pred.max(dim=-1).indices.cpu().numpy())
            train_accs.append(train_acc)
    
        with torch.no_grad(): # Turn off computational graph so as to what the model has seen doesn't affect the weights
            model.eval() # Set model to evaluation mode
            for X_, y_ in iter(test_dl):
                X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                y_p = model(X_)
                test_loss = criterion(y_p, y_)
                
                test_losses.append(test_loss.item())
    
                test_acc = accuracy_score(y_.cpu().numpy(), y_p.max(dim=-1).indices.cpu().numpy())
                test_accs.append(test_acc)
            
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_test_loss = sum(test_losses) / len(test_losses)
    
        avg_train_acc = sum(train_accs) / len(train_accs)
        avg_test_acc = sum(test_accs) / len(test_accs)
    
        print(
            f"Epoch: {epoch+1} | Train loss: {avg_train_loss: .3f} | Test loss: {avg_test_loss: .3f} |",
            f"Train accuracy: {avg_train_acc: .3f} | Test accuracy: {avg_test_acc: .3f}"
        )
    
        TRAIN_LOSSES.append(avg_train_loss)
        TEST_LOSSES.append(avg_test_loss)
    
        TRAIN_ACCS.append(avg_train_acc)
        TEST_ACCS.append(avg_test_acc)

    # Clear CUDA cache
    torch.cuda.empty_cache()
    torch.clear_autocast_cache()

    return {
        "loss": [TRAIN_LOSSES, TEST_LOSSES],
        "accuracy": [TRAIN_ACCS, TEST_ACCS],
        "model": model
    }


# ### Model Training

# Resnet-18 model with finetuning
resnet_model = Model(out_features = NUM_CLASSES, freeze_weights = False, model_function = resnet18).to(DEVICE)

# Intialize model weights
resnet_model = initialize_model_weights(resnet_model, init_func = nn.init.normal_)

# Define optimizer
opt = optim.Adam(
    params = [
        {
            "params": resnet_model.base.fc.parameters(),
            "lr": LR
        }
    ],
    lr = LR/FACTOR,
    amsgrad = AMSGRAD,
    betas=BETAS
)

# Train Resnet-18 via finetuning
resnet_finetuned = training_loop(model = resnet_model, optimizer = opt, epochs = EPOCHS)

# ### Model Testing with ROC AUC Score
def test_model_1(model, data_,):
    train_dl, test_dl = data_

    train_predictions = [
        (y_train.cpu().numpy().reshape(-1, 1), model(X_train.to(DEVICE)).detach().cpu().numpy().reshape(-1, 1))
        for X_train, y_train in iter(train_dl)
    ]
    test_predictions = [
        (y_test.cpu().numpy().reshape(-1, 1), model(X_test.to(DEVICE)).detach().cpu().numpy().reshape(-1, 1))
        for X_test, y_test in iter(test_dl)
    ]

    train_ys = np.concatenate([a for a, b in train_predictions], axis = 0)
    train_preds = np.concatenate([b for a, b in train_predictions], axis = 0)

    train_score = roc_auc_score(train_ys, train_preds, multi_class = "ovo")

    test_ys = np.concatenate([a for a, b in test_predictions], axis = 0)
    test_preds = np.concatenate([b for a, b in test_predictions], axis = 0)

    test_score = roc_auc_score(test_ys, test_preds, multi_class = "ovo")
    # ### Train ROC AUC
    # train_score = [
    #     roc_auc_score(y_train.cpu().numpy().squeeze(), model(X_train.to(DEVICE)).detach().cpu().numpy(), multi_class = "ovr")
    #     for X_train, y_train in iter(train_dl)
    # ]
    # ### Test ROC AUC
    # test_score = [
    #     roc_auc_score(y_test.cpu().numpy().squeeze(), model(X_test.to(DEVICE)).detach().cpu().numpy(), multi_class = "ovr")
    #     for X_test, y_test in iter(test_dl)
    # ]

    # train_score = sum(train_score) / len(train_score)
    # test_score = sum(test_score) / len(test_score)

    return pd.DataFrame(
        data = {
            "Train": [100 * train_score],
            "Test": [100 * test_score],
            "Error (%)": [100 * (train_score - test_score)]
        }
    )

# Accumulate data for testing
data_ = [
    train_dl,
    test_dl
]

# Deploy using ONNX
model = resnet_finetuned['model']  # ORIGINAL nn.Module
# model.eval()

p = next(model.parameters())
print(p.device.type == "cuda")

# Export model for device (Defaulting to CPU for ONNX)
model = model.cpu()

# Create a dummy input tensor with the appropriate shape
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        "./artefacts/models/model.onnx",
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True
    )
print("Model has been converted to ONNX")
