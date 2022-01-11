'''
모든 이미지를 메모리에 로드하여 압축된 Numpy 형식으로 저장

데이터 세트를 로드하고 일부 사진을 플롯하여 이미지 데이터를 확인
'''
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)

# dataset path
path = 'C:/Data/ai.hub/53_sample/'
# load dataset A
dataA1 = load_images(path + 'image/')
dataAB = load_images(path + 'image_test/')
dataA = vstack((dataA1, dataAB))
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB1 = load_images(path + 'night/')
dataB2 = load_images(path + 'night_test/')
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
filename = 'data_set_256.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)


