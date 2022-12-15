import os 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import nibabel as nib

from skimage.util import montage

from albumentations import Compose, HorizontalFlip
# from albumentations.pytorch import ToTensor, ToTensorV2 

def load_image(f_path):
    img = nib.load(f_path)
    img = np.asanyarray(img.dataobj)
    img = np.rot90(img)
    return img

def get_sample_images_info(sample_paths):
    sample_imgs = []
    for path in sample_paths:
        if '_seg' not in path:
            sample_imgs.append(load_image(path))
        else:
            sample_mask = load_image(path)
            mask_WT = sample_mask.copy()
            mask_WT[mask_WT == 1] = 1
            mask_WT[mask_WT == 2] = 1
            mask_WT[mask_WT == 4] = 1

            mask_TC = sample_mask.copy()
            mask_TC[mask_TC == 1] = 1
            mask_TC[mask_TC == 2] = 0
            mask_TC[mask_TC == 4] = 1

            mask_ET = sample_mask.copy()
            mask_ET[mask_ET == 1] = 0
            mask_ET[mask_ET == 2] = 0
            mask_ET[mask_ET == 4] = 1

    print("img shape ->", sample_imgs[0].shape)
    print("mask shape ->", sample_mask.shape)

    #what is the significance of 65th slice?
    fig = plt.figure(figsize=(20, 10))

    gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios=[1, 1.5])
    
    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    flair = ax0.imshow(sample_imgs[0][:,:,65], cmap='bone')
    ax0.set_title("FLAIR", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(flair)

    #  Varying density along a streamline
    ax1 = fig.add_subplot(gs[0, 1])
    t1 = ax1.imshow(sample_imgs[1][:,:,65], cmap='bone')
    ax1.set_title("T1", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t1)

    #  Varying density along a streamline
    ax2 = fig.add_subplot(gs[0, 2])
    t2 = ax2.imshow(sample_imgs[2][:,:,65], cmap='bone')
    ax2.set_title("T2", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t2)

    #  Varying density along a streamline
    ax3 = fig.add_subplot(gs[0, 3])
    t1ce = ax3.imshow(sample_imgs[3][:,:,65], cmap='bone')
    ax3.set_title("T1 contrast", fontsize=18, weight='bold', y=-0.2)
    fig.colorbar(t1ce)

    #  Varying density along a streamline
    ax4 = fig.add_subplot(gs[1, 1:3])

    #ax4.imshow(np.ma.masked_where(mask_WT[:,:,65]== False,  mask_WT[:,:,65]), cmap='summer', alpha=0.6)
    l1 = ax4.imshow(mask_WT[:,:,65], cmap='summer',)
    l2 = ax4.imshow(np.ma.masked_where(mask_TC[:,:,65]== False,  mask_TC[:,:,65]), cmap='rainbow', alpha=0.6)
    l3 = ax4.imshow(np.ma.masked_where(mask_ET[:,:,65] == False, mask_ET[:,:,65]), cmap='winter', alpha=0.6)

    ax4.set_title("", fontsize=20, weight='bold', y=-0.1)

    _ = [ax.set_axis_off() for ax in [ax0,ax1,ax2,ax3, ax4]]

    colors = [im.cmap(im.norm(1)) for im in [l1,l2, l3]]
    labels = ['Non-Enhancing tumor core', 'Peritumoral Edema ', 'GD-enhancing tumor']
    patches = [ mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]

    plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4,fontsize = 'xx-large',
            title='Mask Labels', title_fontsize=18, edgecolor="black",  facecolor='#c5c6c7')
    plt.suptitle("Multimodal Scans -  Data | Manually-segmented mask - Target", fontsize=20, weight='bold')

    fig.savefig("data_sample.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
    fig.savefig("data_sample.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')

    return True 

def get_3D_img_info(dataloader):
    data = next(iter(dataloader))
    data['Id'], data['image'].shape, data['mask'].shape

    img_tensor = data['image'].squeeze()[0].cpu().detach().numpy() 
    mask_tensor = data['mask'].squeeze()[0].squeeze().cpu().detach().numpy()

    print("Num uniq Image values :", len(np.unique(img_tensor, return_counts=True)[0]))
    print("Min/Max Image values:", img_tensor.min(), img_tensor.max())
    print("Num uniq Mask values:", np.unique(mask_tensor, return_counts=True))

    image = np.rot90(montage(img_tensor))
    mask = np.rot90(montage(mask_tensor)) 

    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(image, cmap ='bone')
    ax.imshow(np.ma.masked_where(mask == False, mask),
            cmap='cool', alpha=0.6)
    return True

def get_augmentations(phase):
    list_transforms = []
    
    list_trfms = Compose(list_transforms)
    return list_trfms

