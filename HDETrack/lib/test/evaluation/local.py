from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    # settings.got10k_lmdb_path = ''
    # settings.fe108_path = ''
    # settings.coesot_path = ''
    # settings.visevent_path = ''
    settings.eventvot_path = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/dataset/EventVOT/'
    settings.prj_dir = '/media/amax/c08a625b-023d-436f-b33e-9652dc1bc7c0/DATA/wangshiao/HDETrack'
    settings.result_plot_path = settings.prj_dir + '/output/test/result_plots'
    settings.results_path = settings.prj_dir + '/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = settings.prj_dir + '/output'
    settings.segmentation_path = settings.prj_dir + '/output/test/segmentation_results'
    # settings.tc128_path = ''
    # settings.tn_packed_results_path = ''
    # settings.tnl2k_path = ''
    # settings.tpl_path = ''
    # settings.trackingnet_path = ''
    # settings.uav_path = ''
    # settings.vot18_path = ''
    # settings.vot22_path = ''
    # settings.vot_path = ''
    # settings.youtubevos_dir = ''

    return settings
