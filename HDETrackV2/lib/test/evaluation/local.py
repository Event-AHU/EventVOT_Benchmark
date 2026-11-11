from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.eventvot_path = ''
    settings.prj_dir = ''
    settings.result_plot_path = settings.prj_dir + '/output/test/result_plots'
    settings.results_path = settings.prj_dir + '/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = settings.prj_dir + '/output'


    return settings
