import torch
import torch.nn as nn
from torchvision.models.detection import retinanet

## IMPORTED FROM CRF-NET ##

# Plan:
# 1. locate second head
# 2. replace the head with this file
#    this file contains network (backbone of crf) + fpn + head 
# 3. copy vggmax.py as torch (as backbone)
# 4. copy fpn?
# 5. copy head??
# 6. remove secondary heads and check that everything runs fine

class FusionNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, 
                 dialation=1):
        super(FusionNet, self).__init__()

