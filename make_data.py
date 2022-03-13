import nilearn.image as ni
import glob
from tqdm import tqdm
from scipy.ndimage import zoom
import h5py
import numpy as np

mode='test'
sub_mr_lst = glob.glob('/deneb_disk/headreco_out/'+mode+'/*_T1fs_conform.nii.gz')

patch_size = [256, 256]
X = list()
Y = list()

for sub1 in tqdm(sub_mr_lst):

    sub = sub1[:-20]
    mr = sub + '_T1fs_conform.nii.gz'
    lab = sub + '_masks_contr.nii.gz'

    #print(mr)
    #print(lab)

    mr_data = ni.load_img(mr).get_fdata()
    lab_data = ni.load_img(lab).get_fdata()

    for i in range(mr_data.shape[2]):
        slice = zoom(mr_data[:, :, i], (patch_size[0] /
                     mr_data.shape[0], patch_size[1]/mr_data.shape[1]), order=3)
        labels = zoom(lab_data[:, :, i], (patch_size[0] /
                      mr_data.shape[0], patch_size[1]/mr_data.shape[1]), order=0)
        X.append(slice)
        Y.append(np.uint8(labels))


hf = h5py.File('/deneb_disk/headreco_out/'+mode+'.h5', 'w')
hf.create_dataset('X', data=X)
hf.create_dataset('Y', data=Y)
hf.close()
