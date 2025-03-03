import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN, self).__init__()

        # three convolutional layers to extract features from images
        # kernel size 3x3, stride 1, padding 1
        # stride is similar how the filter or kernel moves across
        # padding is just so we dont miss any pixels on the edges
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) 
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) 

        # max pooling layer to downsample the feature maps
        # reduces the size by half keeps important info
        self.pool = nn.MaxPool2d(2, 2)  

        # fully connected layers to process the extracted features
        # first layer maps the flattened features to 128 neurons
        self.fc1 = nn.Linear(64 * 12 * 12, 128)  

        # final layer outputs the predicted value for age
        self.fc2 = nn.Linear(128, num_classes)  

    def forward(self, x):
        # on the forward pass we use a relu activation
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  

        # flatten the tensor before passing it to fully connected layers
        x = x.view(x.size(0), -1)  

        # apply fully connected layers with relu before final output
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)  

        # squeeze just turns it into a 1D tensor or vector
        return x.squeeze() 
