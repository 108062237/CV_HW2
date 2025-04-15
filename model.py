import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator

def get_faster_rcnn_model(num_classes=11, backbone_name='resnet50', pretrained=True):
    """
    建立 Faster R-CNN 模型。
    Args:
        num_classes (int): 類別數 + 1（包含背景，通常背景是 class 0）
        backbone_name (str): 可選 'resnet50', 'resnet101', 'mobilenet_v2'
        pretrained (bool): 是否使用預訓練權重
    """
    
    if backbone_name.startswith("resnet"):
        # 建立帶有 FPN 的 ResNet 骨幹
        backbone = resnet_fpn_backbone(backbone_name, pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)
    
    elif backbone_name == 'mobilenet_v2':
        # 使用 MobileNetV2 作為骨幹 + FPN
        backbone = torchvision.models.mobilenet_v2(pretrained=pretrained).features
        backbone.out_channels = 1280

        # 建立錨點生成器
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # ROI Pooler（這裡尺寸需與 RPN 一致）
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2
        )

        model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler
        )
    
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    return model
