'''
기능 맵별로 기능이 정규화 되도록 -1로 설정
'''
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
# define layer
layer = InstanceNormalization(axis=-1)

