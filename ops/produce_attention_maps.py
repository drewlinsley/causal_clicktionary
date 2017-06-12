import re
import os
import numpy as np
from glob import glob
from helper_functions import *
from scipy import misc
from matplotlib import pyplot as plt

image_dir = '../all_images'
data_dir = 'alternative_attention_maps/'

attention_type = 'labelme'
image_ext = '.JPEG'
remove_image_wc = 'mircs'

out_dir = 'output/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

fx, file_processor = get_file_extension(attention_type)
image_files = sorted(glob(image_dir + '/*' + image_ext))
proc_image = trim_image_index(image_files,remove_image_wc)
im_size = misc.imread(image_files[0]).shape
image_maps = np.zeros((im_size[0],im_size[1],len(image_files)))
for idx, imp in enumerate(image_files):
	if proc_image[idx]:
		image_name = re.split(image_ext,re.split('/',imp)[-1])[0]
		target_att = data_dir + attention_type + '/' + image_name + fx
		image_maps[:,:,idx] = file_processor(target_att)
	else:
		image_maps[:,:,idx] = np.ones((im_size[0],im_size[1]))
np.savez(out_dir + attention_type,image_maps=image_maps,im_files=image_files)