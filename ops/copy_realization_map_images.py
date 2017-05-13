import shutil
import os
from ops.utilities import make_dir
from glob import glob
from tqdm import tqdm

input_dir = '/media/data_cifs/clicktionary/causal_experiment/clicktionary_probabilistic_region_growth_centered'
output_dir = '/media/data_cifs/clicktionary/webapp_data/clicktionary_probabilistic_region_growth_centered'
im_ext = '.png'

make_dir(output_dir)

folders = glob(os.path.join(input_dir, '*'))
for f in tqdm(folders):
    image_paths = glob(os.path.join(f, '*' + im_ext))
    for im in image_paths:
        shutil.copy2(
            im,
            os.path.join(
                output_dir,
                '%s_%s' % (f.split('/')[-1], im.split('/')[-1]))
            )
