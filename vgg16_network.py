import torch
import torch.nn as nn
import torchvision.models as models

class VGG16_CIFAR10(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16_CIFAR10, self).__init__()
        
        # Load VGG16 with or without pretrained weights
        self.vgg = models.vgg16(pretrained=pretrained)
        
        # Replace the final classifier layer to match CIFAR-10 (10 classes)
        self.vgg.classifier[6] = nn.Linear(4096, 10)

    def forward(self, x):
        return self.vgg(x)

    def get_all(self):
        """Return all layers of the model."""
        return list(self.vgg.children())

# Function to load the pretrained VGG16 model for CIFAR-10
def load_pretrained_vgg16_cifar10(path):
    model = VGG16_CIFAR10(pretrained=False)  # Initialize your model

    # Load the checkpoint
    checkpoint = torch.load(path)

    # Adjust state_dict keys to match the expected ones
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    new_state_dict = {}
    for key in state_dict.keys():
        # Add 'vgg.' prefix to match the expected structure
        new_key = f"vgg.{key}" if not key.startswith("vgg.") else key
        new_state_dict[new_key] = state_dict[key]

    # Load the adjusted state_dict
    model.load_state_dict(new_state_dict, strict=False)  # Use strict=False to ignore minor mismatches
    return model



# Usage example (assuming 'vgg16_cifar10.pth' contains the pretrained model's weights)
if __name__ == "__main__":
    # Load the model and set it to evaluation mode
    model = load_pretrained_vgg16_cifar10("/home/parul/Deep-K-Means-pytorch-master/model/vgg16_cifar10_chkpts.pth")
    model.eval()  # Important for inference

    # Print model structure for verification
    print(model)
