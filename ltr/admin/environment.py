import importlib
import os
from collections import OrderedDict


def create_default_local_file():
    path = os.path.join(os.path.dirname(__file__), 'local.py')

    empty_str = '\'\''
    default_settings = OrderedDict({
        'rgbnirval_dir': '/data/RGBNIR',
        'rgbnir_dir': '/data/RGBNIR/',
        'rgbnir_rgb_dir': '/data/RGBNIR_RGB/',
        'rgbnir_nir_dir': '/data/RGBNIR_NIR/',

        'rgbnirplusval_dir': '/data/RGBNIR_PLUS',
        'rgbnirplus_dir': '/data/RGBNIR_PLUS/',
        'rgbnirplus_rgb_dir': '/data/RGBNIR_PLUS_RGB/',
        'rgbnirplus_nir_dir': '/data/RGBNIR_PLUS_NIR/',
        
        
        'rgbt_dir': '/data/RGBT/',
        'rgbt_rgb_dir': '/data/RGBT_RGB/',
        'rgbt_t_dir': '/data/RGBT_T/',

        'workspace_dir': empty_str,
        'tensorboard_dir': 'self.workspace_dir + \'/tensorboard/\'',
        'pretrained_networks': 'self.workspace_dir + \'/pretrained_networks/\'',
        'lasot_dir': empty_str,
        'got10k_dir': empty_str,
        'trackingnet_dir': empty_str,
        'coco_dir': empty_str,
        'lvis_dir': empty_str,
        'sbd_dir': empty_str,
        'imagenet_dir': empty_str,
        'imagenetdet_dir': empty_str,
        'ecssd_dir': empty_str,
        'hkuis_dir': empty_str,
        'msra10k_dir': empty_str,
        'davis_dir': empty_str,
        'youtubevos_dir': empty_str})

    comment = {'workspace_dir': 'Base directory for saving network checkpoints.',
               'tensorboard_dir': 'Directory for tensorboard files.'}

    with open(path, 'w') as f:
        f.write('class EnvironmentSettings:\n')
        f.write('    def __init__(self):\n')

        for attr, attr_val in default_settings.items():
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            if comment_str is None:
                f.write('        self.{} = {}\n'.format(attr, attr_val))
            else:
                f.write('        self.{} = {}    # {}\n'.format(attr, attr_val, comment_str))


def env_settings():
    env_module_name = 'ltr.admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. Then try to run again.'.format(env_file))
