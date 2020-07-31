import os
import glob
import json
import numpy as np

np.random.seed(6)


ROOT = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT, 'data', 'distributed')
SPLIT_CFG_DIR = os.path.join(ROOT, 'data', 'cfgs')
DATASET_INFO_CFG = os.path.join(SPLIT_CFG_DIR, 'dataset_info.json')
INS_SPLIT_CFG = os.path.join(SPLIT_CFG_DIR, 'split_by_ins.json')

for i in [os.path.join(ROOT, 'data'), SAVE_DIR, SPLIT_CFG_DIR]:
    if not os.path.exists(i):
        os.mkdir(i)


def _load_dataset_info():
    with open(DATASET_INFO_CFG, 'r') as fp:
        f = json.load(fp)
    return f

def _generate_data_info(data_dir):
    images = glob.glob(os.path.join(data_dir, 'image') + os.sep + '*.npy')
    ins_to_patient_id = dict()
    for i in images:
        # Brats18_CBICA_AQT_1_<slice_no>.npy
        patient_id = os.path.basename(i)
        # Brats18_CBICA_AQT_1_<slice_no>
        patient_id = patient_id[:patient_id.rfind('.npy')]
        item = patient_id.split('_')
        ins = item[1]
        if ins not in ins_to_patient_id:
            ins_to_patient_id[ins] = list()
        # real
        pid = '_'.join(item[:4])
        if pid not in ins_to_patient_id[ins]:
            ins_to_patient_id[ins].append(pid)
    with open(DATASET_INFO_CFG, 'w') as fp:
        json.dump({'ins_to_patient_id': ins_to_patient_id}, fp)


def _get_data_files(img_glob_regex, lbl_glob_regex):
    images = glob.glob(img_glob_regex)
    labels = glob.glob(lbl_glob_regex)
    for i, l in zip(images, labels):
        assert os.path.basename(i) == os.path.basename(l)
    return images, labels

def _get_data_files_by_patient_id(data_dir, pid):
    img_regex = os.path.join(data_dir, 'image', pid + '*.npy')
    lbl_regex = os.path.join(data_dir, 'mask', pid + '*.npy')
    return _get_data_files(img_regex, lbl_regex)


def split_by_insitiution(data_dir, train_rate, val_rate, train_test_val_no_cross=True):
    """
    按照结构划分数据, 保存为 config
    :param packs
    :return:
    """
    cfg = _load_dataset_info()
    res = {}

    for ins, ins_pat_name in cfg['ins_to_patient_id'].items():
        # ins_pat_name 不含 slice_no
        print('Process: ', ins, end=' ')
        if train_test_val_no_cross:
            ds = len(ins_pat_name)
            pre = np.random.permutation(ds)
            all_pids = np.asarray(ins_pat_name)
        else:
            raise NotImplementedError
        train_ps = all_pids[pre[:int(train_rate * ds)]]
        if val_rate != 0.0:
            val_ps = all_pids[pre[len(train_ps):len(train_ps) + int(val_rate * ds)]]
            test_ps = all_pids[pre[len(train_ps) + len(val_ps):]]
        else:
            test_ps = all_pids[pre[len(train_ps):]]
            val_ps = None
        # 转换为对应的文件名
        output = dict()
        train_image_fns, train_mask_fns = [], []
        for x in train_ps:
            i, l = _get_data_files_by_patient_id(data_dir, x)
            train_image_fns.extend(i)
            train_mask_fns.extend(l)
        output['train'] = {'image': train_image_fns, 'mask': train_mask_fns}
        if val_rate != 0.0:
            val_image_fns, val_mask_fns = [], []
            for x in val_ps:
                i, l = _get_data_files_by_patient_id(data_dir, x)
                val_image_fns.extend(i)
                val_mask_fns.extend(l)
            output['val'] = {'image': val_image_fns, 'mask': val_mask_fns}
        test_image_fns, test_mask_fns = [], []
        for x in test_ps:
            i, l = _get_data_files_by_patient_id(data_dir, x)
            test_image_fns.extend(i)
            test_mask_fns.extend(l)
        output['test'] = {'image': test_image_fns, 'mask': test_mask_fns}
        #
        split_path = os.path.join(SAVE_DIR, f'train_{int(train_rate * 100)}_test_{100 - int((train_rate + val_rate) * 100)}')
        if not os.path.exists(split_path):
            os.mkdir(split_path)
        with open(os.path.join(split_path, ins + '.json'), 'w') as fp:
            json.dump(output, fp)
        if val_rate != 0.0:
            print('Train size: ', len(train_mask_fns), ', test size: ', len(test_image_fns), ', val size: ', len(val_mask_fns), end=' ')
            print('TRAIN PIDS:', train_ps, 'TEST PID:', test_ps, 'VAL PIDS: ', val_ps)
        else:
            print('Train size: ', len(train_mask_fns), ', test size: ', len(test_image_fns))
            print('TRAIN PIDS:', train_ps, 'TEST PID:', test_ps)


def split_by_merged_insitiution(splitted_cfgs_dir, subname, merged_list):
    """
    按照结构划分数据, 但将机构合并
    :param packs
    :return:
    """
    split_path = os.path.join(SAVE_DIR, f'merged_' + subname)
    if not os.path.exists(split_path):
        os.mkdir(split_path)
    # merged_list : [[1,2,3],[4,5,6]...]
    for merged in merged_list:
        # merged = [1,2,3]
        name_jsons = []
        for filenames in merged:
            with open(os.path.join(splitted_cfgs_dir, filenames + '.json')) as fp:
                name_jsons.append(json.load(fp))
        new = dict()
        # 合并相关
        for info in name_jsons:
            # {'train': [], 'test': []}
            for k, v in info.items():
                if k not in new:
                    new[k] = {'image': [], 'mask': []}
                new[k]['image'].extend(v['image'])
                new[k]['mask'].extend(v['mask'])
        #
        if 'val' in new:
            print('Train size: ', len(new['train']['mask']), ', test size: ', len(new['test']['mask']), ', val size: ',
                  len(new['val']['mask']))
        else:
            print('Train size: ', len(new['train']['mask']), ', test size: ', len(new['test']['mask']))
        path = os.path.join(split_path, '_'.join(merged) + '.json')
        with open(path, 'w') as fp:
            json.dump(new, fp)

if __name__ == '__main__':
    # data_dir = '/home/liuyuan/shu_codes/pytorch_brats/data/brats2018_train'
    # _generate_data_info(data_dir)
    # split_by_insitiution(data_dir=data_dir, train_rate=0.8, val_rate=0.0, train_test_val_no_cross=True)
    split_by_merged_insitiution('data/distributed/train_80_test_20', '3_ins', [
        ['CBICA'], ['TCIA13', 'TCIA02', '2013', 'TCIA01'],
        ['TCIA09', 'TCIA12', 'TCIA10', 'TCIA04', 'TCIA06', 'TCIA03', 'TCIA05']
    ])