import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a basic CNN block to train over CIFAR10
class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.pool(out)
        return out

# Define the full CNN similar to WideResNet structure
class CNN_Network(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_Network, self).__init__()
        self.in_channels = 3  # Updated to 3 for CIFAR-10 RGB images

        # Define the different stages (similar to WideResNet's stages)
        self.stage1 = self._make_stage(16, stride=1)
        self.stage2 = self._make_stage(16, stride=1)
        self.stage3 = self._make_stage(32, stride=1)
        self.stage4 = self._make_stage(32, stride=1)

        # A final fully connected layer (updated input features to 128)
        self.fc = nn.Linear(32 * 2 * 2, num_classes)

    def _make_stage(self, out_channels, stride):
        # Create a single stage consisting of a CNN block
        stage = CNN_block(self.in_channels, out_channels, stride)
        self.in_channels = out_channels
        return stage

    def forward(self, x):
        x = self.stage1(x)  # -> [batch, 16, 16, 16]
        x = self.stage2(x)  # -> [batch, 16, 8, 8]
        x = self.stage3(x)  # -> [batch, 32, 4, 4]
        x = self.stage4(x)  # -> [batch, 32, 2, 2]
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.fc(x)
        return x

# Instantiate and print the model to verify
if __name__ == "__main__":
    model = CNN_Network(num_classes=10)
    print(model)
