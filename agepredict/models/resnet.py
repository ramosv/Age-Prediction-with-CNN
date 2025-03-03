import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class RestNet(nn.Module):
    # num classes 1 for regression
    def __init__(self, num_classes=1, pretrained=True):
        super(RestNet, self).__init__()

        # at the end when we want to predict the age we set weights to None
        # becase we are using the weights from the pretrained model
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
            
        self.model = models.resnet18(weights=weights)
        
        # replacing the fully connected layer with a dropout layer followed by a linear layer
        # dropout randomly turns off a percentage of neurons during training 30% below
        # linear layer outputs the final age prediction
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.fc.in_features, num_classes)
        )
    
    # forward pass
    def forward(self, x):
        return self.model(x).squeeze()
