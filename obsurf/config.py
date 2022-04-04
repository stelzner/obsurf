import yaml
from torchvision import transforms
from obsurf import data
from obsurf import model

import os


CFG = None


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.CLoader)


    # If no, use the default_path
    if default_path is not None:
        with open(default_path, 'r') as f:
            cfg_default = yaml.load(f, Loader=yaml.CLoader)
    else:
        cfg_default = dict()

    cfg_import = collect_imports(cfg_special)
    print('IMPORTS')
    print(cfg_import)

    # Include main configuration
    update_default(cfg_special, cfg_import)
    update_default(cfg_special, cfg_default)

    print('FINAL CONFIG')
    print(cfg_special)
    global CFG
    CFG = cfg_special
    return cfg_special


def collect_imports(d):
    base_dict = {}
    for k, v in d.items():
        if k == 'import':
            with open(v, 'r') as f:
                cfg_import = yaml.load(f, Loader=yaml.CLoader)
            for k_i, v_i in cfg_import.items():
                if k_i not in base_dict:
                    base_dict[k_i] = v_i
        elif isinstance(v, dict):
            base_dict[k] = collect_imports(v)
    return base_dict


def update_default(dict_special, dict_default):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary containing default entries

    '''
    for k, v in dict_default.items():
        if isinstance(v, dict):
            if k not in dict_special:
                dict_special[k] = dict()
            update_default(dict_special[k], v)
        else:
            if k not in dict_special:
                dict_special[k] = v


# Datasets
def get_dataset(mode, cfg, return_idx=False, return_category=False, dummy=False, max_len=None, max_views=None):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset']
    dataset_folder = cfg['data']['path']
    jitter = 'jitter' in cfg['data'] and cfg['data']['jitter']
    augment = 'augment' in cfg['data'] and cfg['data']['augment']
    points_per_item = cfg['data']['num_points'] if 'num_points' in cfg['data'] else 2048
    importance_cutoff = cfg['data']['importance_cutoff'] if 'importance_cutoff' in cfg['data'] else 0.98

    # Create dataset
    if mode == 'real':
        dataset = data.ClevrRealDataset('./data/clevr-real/')
    elif dataset_type == 'sprites':
        variant = cfg['data']['variant']
        if dummy:
            path = os.path.join(dataset_folder, f'{variant}-dummy.npz')
        else:
            path = os.path.join(dataset_folder, f'{variant}-{mode}.npz')
        dataset = data.SpriteDataset(path, mode, max_len=max_len)
    elif dataset_type == 'mnist':
        dataset = data.MnistDataset(dataset_folder, mode, jitter=jitter)
    elif dataset_type == 'clevr2d':
        dataset = data.Clevr2dDataset(dataset_folder, mode, jitter=jitter, max_len=max_len)
    elif dataset_type == 'multishapenet2d':
        dataset = data.Clevr2dDataset(dataset_folder, mode, jitter=jitter, shapenet=True)
    elif dataset_type == 'clevr3d':
        dataset = data.Clevr3dDataset(dataset_folder, mode, points_per_item=points_per_item,
                                      shapenet=False, max_len=max_len, max_views=max_views,
                                      importance_cutoff=importance_cutoff)
    elif dataset_type == 'multishapenet':
        dataset = data.Clevr3dDataset(dataset_folder, mode, points_per_item=points_per_item,
                                      shapenet=True, max_len=max_len, max_views=max_views,
                                      importance_cutoff=importance_cutoff)
    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset'])

    return dataset


