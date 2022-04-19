import numpy as np
from tqdm import tqdm
from utils import test_images, test_single_nii, calculate_metric_percase
from data_reader import H5DataLoader
import h5py

data_file = 'test_t1t2.h5'

predictions_file = 'test_t1t2_output.h5'
#predictions_file = 'test_t1.h5'


test = h5py.File(data_file, 'r')['Y']

pred = h5py.File(predictions_file, 'r')['labels']

num_classes = 9
num_imgs = 1000#test.shape[0]
dice_coeffs = np.zeros([num_imgs,num_classes])
hausdorff_dist = np.zeros([num_imgs,num_classes])


for i in tqdm(range(num_imgs)):

    metric_list = []
    for lid in range(num_classes):
        dice_coeffs[i,lid], hausdorff_dist[i,lid] = calculate_metric_percase(test[i] == lid, pred[i] == lid)


print(np.nanmean(dice_coeffs,axis=0)) #, np.nanstd(dice_coeffs,axis=0))


    

