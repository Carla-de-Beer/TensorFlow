"""
Enums.
"""

import enum


class LandscapeType(enum.Enum):
    desert = 'desert'
    forest = 'forest'
    polar = 'polar'


class ImageInfo(enum.Enum):
    size = 150


class NetworkParams(enum.Enum):
    accuracy = 0.95
    num_epochs = 10
