import numpy as np
import glob
import os
import sys
import nibabel as nib  # nii格式一般都会用到这个包

flair_name = "_flair.nii.gz"
t1_name = "_t1.nii.gz"
t1ce_name = "_t1ce.nii.gz"
t2_name = "_t2.nii.gz"
mask_name = "_seg.nii.gz"

'''
SEG:
0: other
1: necrosis + non-enhancing tumor
2: edema
4: enhancing tumor
5: full tumor
'''
"""
MRI:
whole/complete tumor: 1 2 4
core: 1 4
enhance: 4
"""

def nii_to_image(image_niifile, label_niifile, mask_dir, image_dir, image_name, include_mask=True, mask_label=5):
    img = nib.load(image_niifile).get_fdata()  # 读取nii
    if include_mask:
        label = nib.load(label_niifile).get_fdata()
        # 开始转换为图像
        # (H, W, SLICE)
        # 处理 label图像
        if mask_label == 5:
            label[label != 0] = 1  # Region 1 => 1+2+3+4 complete tumor
        if mask_label == 1:
            label[label != 1] = 0  # only left necrosis
        if mask_label == 2:
            label[label == 2] = 0  # turn edema to 0
            label[label != 0] = 1  # only keep necrosis, ET, NET = Tumor core
        if mask_label == 4:
            label[label != 4] = 0  # only left ET
            label[label == 4] = 1
        label = label.astype(np.float32)
    # 处理图像
    # 输入图像, 可以设置其他的切片
    img = (img - img.mean()) / img.std()  # normalization => zero mean   !!!care for the std=0 problem
    img = img.astype('float32')  # 图像使用 float32
    if include_mask:
        for i in range(img.shape[2]):
            np.save(file=os.path.sep.join((mask_dir, '{0}_{1}.npy'.format(image_name, i))), arr=label[:, :, i])
            np.save(file=os.path.sep.join((image_dir, '{0}_{1}.npy'.format(image_name, i))), arr=img[:, :, i])
    else:
        for i in range(img.shape[2]):
            np.save(file=os.path.sep.join((image_dir, '{0}_{1}.npy'.format(image_name, i))), arr=img[:, :, i])

def convert_dataset(prefix, target_dir, max_size=None):
    tts = [target_dir, os.sep.join((target_dir, 'ground_truth')), os.sep.join((target_dir, 'flair'))]
    for tt in tts:
        if not os.path.exists(tt):
            os.makedirs(tt, exist_ok=True)

    # 得到所有的 nii 文件
    subset_paths = glob.glob(prefix + '{0}**'.format(os.sep), recursive=False)[:max_size]

    for i, subset_path in enumerate(subset_paths):
        subset_name = os.path.split(subset_path)[-1]
        subset_data_prefix = '{0}{1}{2}'.format(subset_path, os.sep, subset_name)
        flair_image = subset_data_prefix + flair_name
        seg_image = subset_data_prefix + mask_name
        nii_to_image(image_niifile=flair_image, label_niifile=seg_image, mask_dir=tts[-2], image_dir=tts[-1], image_name=subset_name)


def convert_validation_dataset(prefix, target_dir, max_size=None):
    tts = [target_dir, os.sep.join((target_dir, 'flair'))]
    for tt in tts:
        if not os.path.exists(tt):
            os.mkdir(tt)

    # 得到所有的 nii 文件
    subset_paths = glob.glob(prefix + '{0}**'.format(os.sep), recursive=False)[:max_size]

    for i, subset_path in enumerate(subset_paths):
        subset_name = os.path.split(subset_path)[-1]
        subset_data_prefix = '{0}{1}{2}'.format(subset_path, os.sep, subset_name)
        flair_image = subset_data_prefix + flair_name
        nii_to_image(image_niifile=flair_image, label_niifile=None, mask_dir=None, image_dir=tts[1], image_name=subset_name, include_mask=False)




if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python {} <in_path> <out_path>'.format(sys.argv[0]))
        exit(1)
    else:
        filepath = sys.argv[1]
        out = sys.argv[2]
    convert_dataset(prefix=filepath, target_dir='data\\origin', max_size=None)

    # convert_validation_dataset(prefix='data\\MICCAI_BraTS_2018_Data_Validation', target_dir='data\\brats2018\\test')
    # for i in range(20, 22):
    #     if i == 15 or i == 16:
    #         continue
    #     check_image('data\\brats2018\\training', 'Brats18_2013_{}_1'.format(i))