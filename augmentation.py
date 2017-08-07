import numpy as np

def augmentation(img, aug_horz_max, aug_vert_max, aug_noise_max, aug_const_max):
    img_size = img.shape

    img_bg = np.empty_like(img)
    aug_horz = np.empty_like(img)
    aug_vert = np.empty_like(img)
    aug_noise = np.empty_like(img)
    img_aug = np.empty_like(img)

    # pattern 1: horizontal gradient
    aug_horz_end = np.random.uniform(-aug_horz_max, aug_horz_max)
    aug_horz_slop = aug_horz_end / (img_size[1]/2.)
    # pattern 2: vertical gradient
    aug_vert_end = np.random.uniform(-aug_vert_max, aug_vert_max)
    aug_vert_slop = aug_vert_end / (img_size[0]/2.)
    # pattern 4: const
    aug_const = np.random.uniform(-aug_const_max, aug_const_max)
    
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            aug_horz[i,j] = aug_horz_slop*(j-img_size[1]/2.)
            aug_vert[i,j] = aug_vert_slop*(i-img_size[0]/2.)
            aug_noise[i,j] = np.random.uniform(-aug_noise_max, aug_noise_max)
            bg = aug_horz[i,j] + aug_vert[i,j] + aug_noise[i,j] + aug_const
            if(img[i,j] == 0):
                img_aug[i,j] = bg
            else:
                img_aug[i,j] = img[i,j] - img[img_size[0]/2, img_size[1]/2]
    img_aug = img_aug.reshape((img_aug.shape[0], img_aug.shape[1], 1))
    return img_aug