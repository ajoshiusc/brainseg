import numpy as np
from tqdm import tqdm
from utils import test_images, test_single_nii, calculate_metric_percase
from data_reader import H5DataLoader


data_file = 'test_t1t2.h5'

predictions_file = 'test_t1t2_output.h5'

test = H5DataLoader(data_file)

pred = H5DataLoader(predictions_file)

num_classes = 9
num_imgs = test.images.shape[0]
dice_coeffs = np.zeros([num_imgs,num_classes])
hausdorff_dist = np.zeros([num_imgs,num_classes])


for i in range(num_imgs):

    metric_list = []
    for lid in range(num_classes):
        dice_coeffs[i,lid], hausdorff_dist[i,lid] = calculate_metric_percase(test.labels[i] == lid, pred.labels[i] == lid)



    

