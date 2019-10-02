from ..registry import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module
class RetinaNet(SingleStageDetector):
    """
    RetinaNet is a one-stage detector that uses a feature pyramid network on
    top of a downsampling backbone.

    .. _"Focal loss for dense object detection." ICCV. 2017:
        https://arxiv.org/abs/1708.02002
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
