# Այստեղ տեղադրիր model.py-ի քո ամբողջական կոդը՝ ներառյալ նոր unfreeze_last_block ֆունկցիան

import torch
import torch.nn as nn

def unfreeze_last_block(model, num_blocks=2):
    """
    Ավտոմատ որոշում է մոդելի տեսակը (ResNet կամ EfficientNet)
    և ապասառեցնում է վերջին բլոկները fine-tuning-ի համար:
    """
    if hasattr(model, "net"):
        target_model = model.net
    elif hasattr(model, "resnet"):
        target_model = model.resnet
    else:
        target_model = model

    # 1. Եթե ResNet է
    if hasattr(target_model, "layer4"):
        for param in target_model.layer4.parameters():
            param.requires_grad = True
        if hasattr(target_model, "fc"):
            for param in target_model.fc.parameters():
                param.requires_grad = True
        print("🔓 ResNet layer4-ը և fc-ն ապաբլոկավորվեցին fine-tuning-ի համար:")

    # 2. Եթե EfficientNet է
    elif hasattr(target_model, "features"):
        features_module = target_model.features
        for block in list(features_module.children())[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(target_model, "classifier"):
            for param in target_model.classifier.parameters():
                param.requires_grad = True
        print(f"🔓 EfficientNet-ի վերջին {num_blocks} բլոկները և classifier-ը ապաբլոկավորվեցին fine-tuning-ի համար:")

    else:
        print("⚠️ Չհաջողվեց գտնել ապասառեցման ենթակա շերտերը:")

    return model

# (Այստեղ ավելացրու նաև EfficientNetV2SCassava, ResNet, և մյուս դասերը, որոնք ունես model.py-ում)

def unfreeze_last_block(model, num_blocks=2):
    """
    Ավտոմատ որոշում է մոդելի տեսակը (ResNet կամ EfficientNet)
    և ապասառեցնում է վերջին բլոկները fine-tuning-ի համար:
    """
    if hasattr(model, "net"):
        target_model = model.net
    elif hasattr(model, "resnet"):
        target_model = model.resnet
    else:
        target_model = model

    # 1. Եթե ResNet է
    if hasattr(target_model, "layer4"):
        for param in target_model.layer4.parameters():
            param.requires_grad = True
        if hasattr(target_model, "fc"):
            for param in target_model.fc.parameters():
                param.requires_grad = True
        print("🔓 ResNet layer4-ը և fc-ն ապաբլոկավորվեցին fine-tuning-ի համար:")

    # 2. Եթե EfficientNet է
    elif hasattr(target_model, "features"):
        features_module = target_model.features
        for block in list(features_module.children())[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        if hasattr(target_model, "classifier"):
            for param in target_model.classifier.parameters():
                param.requires_grad = True
        print(f"🔓 EfficientNet-ի վերջին {num_blocks} բլոկները և classifier-ը ապաբլոկավորվեցին fine-tuning-ի համար:")

    else:
        print("⚠️ Չհաջողվեց գտնել ապասառեցման ենթակա շերտերը:")

    return model

# (Այստեղ ավելացրու նաև EfficientNetV2SCassava, ResNet, և մյուս դասերը, որոնք ունես model.py-ում)
