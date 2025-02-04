import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils import load_data, create_dataloaders, get_transforms

# Data loading and preprocessing
data_dir = 'data/logos/LogoDet-3K'
transform = get_transforms()
train_dataset, val_dataset = load_data(data_dir, transform)
train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, batch_size=32)

# Model definition, criterion, and optimizer
model_classifier = models.resnet50(pretrained=True)
model_classifier.fc = nn.Linear(model_classifier.fc.in_features, len(train_dataset.dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_classifier.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_classifier.to(device) # Move model to device

for epoch in range(num_epochs):
    model_classifier.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device) # Move data to device
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model_classifier(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model_classifier.state_dict(), 'models/logo_classifier.pth')
