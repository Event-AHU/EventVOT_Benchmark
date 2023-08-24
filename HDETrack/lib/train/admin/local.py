class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'
        # self.coesot_dir = ''
        # self.coesot_val_dir = ''
        # self.fe108_dir = ''
        # self.fe108_val_dir = ''
        # self.visevent_dir = ''
        # self.visevent_val_dir = ''
        self.eventvot_dir = ''
        self.eventvot_val_dir = ''