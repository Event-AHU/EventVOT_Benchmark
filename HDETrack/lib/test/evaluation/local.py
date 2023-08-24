from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    # settings.fe108_path = ''
    settings.eventvot_path = ''
    settings.prj_dir = ''
    settings.result_plot_path = settings.prj_dir + '/output/test/result_plots'
    settings.results_path = settings.prj_dir + '/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = settings.prj_dir + '/output'
    settings.segmentation_path = settings.prj_dir + '/output/test/segmentation_results'

    return settings
