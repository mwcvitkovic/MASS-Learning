from .SmallMLP import SmallMLP
from .ResNet20 import ResNet20


class NullNet():
    def __init__(self, *args, **kwargs):
        self.out_dim = kwargs['out_dim']
