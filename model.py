import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import MultiScaleRoIAlign

def get_faster_rcnn_model(num_classes=11, backbone_name='resnet50', pretrained=True):
    
    
    if backbone_name.startswith("resnet"):
        # 建立帶有 FPN 的 ResNet 骨幹
        backbone = resnet_fpn_backbone(backbone_name, pretrained=pretrained)
        model = FasterRCNN(backbone, num_classes=num_classes)

    elif backbone_name == 'resnext101_32x8d':
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


def build_fpn_model_from_backbone(backbone_cnn, num_classes):
    # Feature extractor：抽出要給 FPN 的中間層
    return_layers = {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3',
    }

    in_channels_list = [256, 512, 1024, 2048]  # 對應每層的 channel 數
    out_channels = 256                         # FPN 每層輸出的 channel 數

    # 建立 feature extractor
    feature_extractor = create_feature_extractor(backbone_cnn, return_nodes=return_layers)

    # 正確初始化 BackboneWithFPN
    backbone_with_fpn = BackboneWithFPN(
        backbone=feature_extractor,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=out_channels,
        extra_blocks=LastLevelMaxPool()
    )

    # 建立 Faster R-CNN 模型
    model = torchvision.models.detection.FasterRCNN(
        backbone=backbone_with_fpn,
        num_classes=num_classes
    )

    return model