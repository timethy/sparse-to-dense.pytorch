import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

cmap = plt.cm.viridis

def merge_into_row(input, target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth = np.squeeze(target.cpu().numpy())
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:,:,:3] # H, W, C
    pred = np.squeeze(depth_pred.data.cpu().numpy())
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred = 255 * cmap(pred)[:,:,:3] # H, W, C
    img_merge = np.hstack([rgb, depth, pred])
    
    # img_merge.save(output_directory + '/comparison_' + str(epoch) + '.png')
    return img_merge

def merge_into_row(input, depth_sparse, target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1,2,0)) # H, W, C
    depth_sparse = np.squeeze(depth_sparse.cpu().numpy())
    depth_sparse = (depth_sparse - np.min(depth_sparse)) / (np.max(depth_sparse) - np.min(depth_sparse))
    depth_sparse = 255 * cmap(depth_sparse)[:,:,:3] # H, W, C
    depth = np.squeeze(target.cpu().numpy())
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:,:,:3] # H, W, C
    pred = np.squeeze(depth_pred.data.cpu().numpy())
    pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    pred = 255 * cmap(pred)[:,:,:3] # H, W, C
    img_merge = np.hstack([rgb, depth_sparse, depth, pred])

    # img_merge.save(output_directory + '/comparison_' + str(epoch) + '.png')
    return img_merge

def add_row(img_merge, row):
    return np.vstack([img_merge, row])

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)