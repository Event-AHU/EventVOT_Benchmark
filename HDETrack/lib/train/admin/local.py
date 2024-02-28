class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/wangshiao/HDETrack'  # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'
        self.eventvot_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/EventVOT/train'
        # self.lvis_dir = ''
        # self.sbd_dir = ''
        # self.imagenetdet_dir = ''
        # self.ecssd_dir = ''
        # self.hkuis_dir = ''
        # self.msra10k_dir = ''
        # self.davis_dir = ''
        # self.youtubevos_dir = ''
        # self.coesot_dir = ''
        # self.coesot_val_dir = ''
        # self.fe108_dir = ''
        # self.fe108_val_dir = ''
        # self.visevent_dir = ''
        # self.visevent_val_dir = ''