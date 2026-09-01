import torch
import torch.nn as nn
import torchvision.models as models

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu4(self.fc1(x)))
        x = self.fc2(x)
        return x


class DINOv2Cassava(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=False):
        super(DINOv2Cassava, self).__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        in_features = self.backbone.embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out


class EfficientNetV2SCassava(nn.Module):
    def __init__(self, num_classes=5, model_name='efficientnet_b2', freeze_backbone=False):
        super(EfficientNetV2SCassava, self).__init__()
        
        if model_name == 'efficientnet_b2':
            weights = models.EfficientNet_B2_Weights.DEFAULT
            self.backbone = models.efficientnet_b2(weights=weights)
        else:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.backbone = models.efficientnet_b0(weights=weights)
            
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class ResNet50Cassava(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True):
        super(ResNet50Cassava, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def unfreeze_last_block(model, num_blocks=2):
    
    if hasattr(model, "net"):
        target_model = model.net
    elif hasattr(model, "resnet"):
        target_model = model.resnet
    else:
        target_model = model

    #  Եթե ResNet է
    if hasattr(target_model, "layer4"):
        for param in target_model.layer4.parameters():
            param.requires_grad = True
        if hasattr(target_model, "fc"):
            for param in target_model.fc.parameters():
                param.requires_grad = True
        print(" ResNet layer4-ը և fc-ն ապաբլոկավորվեցին fine-tuning-ի համար:")

    #  Եթե EfficientNet է
    elif hasattr(target_model, "features"):
        features_module = target_model.features
        for block in list(features_module.children())[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(target_model, "classifier"):
            for param in target_model.classifier.parameters():
                param.requires_grad = True
        print(f" EfficientNet-ի վերջին {num_blocks} բլոկները և classifier-ը ապաբլոկավորվեցին fine-tuning-ի համար:")

    else:
        print(" Չհաջողվեց գտնել ապասառեցման ենթակա շերտերը:")

    return model
