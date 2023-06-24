import torch.nn as nn
import torchvision

def efficient_net(out_features, version='b0'):
    model = torchvision.models.get_model('efficientnet_' + version.lower(), weights='DEFAULT')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, out_features, bias=True)
    for param in model.features[:-3].parameters():
        param.requires_grad = False
    model.transforms = torchvision.models.get_weight(f'EfficientNet_{version.upper()}_Weights.DEFAULT').transforms()
    return model
