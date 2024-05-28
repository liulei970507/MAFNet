class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data/liulei/RGBNIR/DIMP_AL/ltr/'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data/liulei/RGBNIR/DIMP_AL/ltr/tensorboard/'    # Directory for tensorboard files.
        self.rgbnir_dir = '/data/RGBNIR/'
        self.rgbnir_rgb_dir = '/data/RGBNIR_RGB/'
        self.rgbnir_nir_dir = '/data/RGBNIR_NIR/'

        self.rgbnirplus_dir = '/data/RGBNIR_PLUS/'
        self.rgbnirplus_rgb_dir = '/data/RGBNIR_PLUS_RGB/'
        self.rgbnirplus_nir_dir = '/data/RGBNIR_PLUS_NIR/'
        
        
        
        self.rgbt_dir = '/data/RGBT/'
        self.rgbt_rgb_dir = '/data/RGBT_RGB/'
        self.rgbt_t_dir = '/data/RGBT_T/'