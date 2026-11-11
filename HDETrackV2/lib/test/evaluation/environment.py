import importlib
import os


class EnvSettings:
    def __init__(self):
        test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.results_path = '{}/tracking_results/'.format(test_path)
        self.segmentation_path = '{}/segmentation_results/'.format(test_path)
        self.network_path = '{}/networks/'.format(test_path)
        self.result_plot_path = '{}/result_plots/'.format(test_path)
        self.coesot_path =  ''
        self.eventvot_path =  ''
        self.fe108_path =  ''
        self.visevent_path =  ''


def create_default_local_file():
    comment = {'results_path': 'Where to store tracking results',
               'network_path': 'Where tracking networks are stored.'}

    path = os.path.join(os.path.dirname(__file__), 'local.py')
    with open(path, 'w') as f:
        settings = EnvSettings()

        f.write('from test.evaluation.environment import EnvSettings\n\n')
        f.write('def local_env_settings():\n')
        f.write('    settings = EnvSettings()\n\n')
        f.write('    # Set your local paths here.\n\n')

        for attr in dir(settings):
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            attr_val = getattr(settings, attr)
            if not attr.startswith('__') and not callable(attr_val):
                if comment_str is None:
                    f.write('    settings.{} = \'{}\'\n'.format(attr, attr_val))
                else:
                    f.write('    settings.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))
        f.write('\n    return settings\n\n')


class EnvSettings_ITP:
    def __init__(self, workspace_dir, data_dir, save_dir):
        self.prj_dir = workspace_dir
        self.save_dir = save_dir
        self.results_path = os.path.join(save_dir, 'test/tracking_results')
        self.segmentation_path = os.path.join(save_dir, 'test/segmentation_results')
        self.network_path = os.path.join(save_dir, 'test/networks')
        self.result_plot_path = os.path.join(save_dir, 'test/result_plots')
        self.coesot_path = os.path.join(data_dir, 'COESOT')
        self.eventvot_path = os.path.join(data_dir, 'EventVOT')
        self.fe108_path = os.path.join(data_dir, 'FE108')
        self.visevent_path = os.path.join(data_dir, 'VisEvent')


def create_default_local_file_ITP_test(workspace_dir, data_dir, save_dir):
    comment = {'results_path': 'Where to store tracking results',
               'network_path': 'Where tracking networks are stored.'}

    path = os.path.join(os.path.dirname(__file__), 'local.py')
    with open(path, 'w') as f:
        settings = EnvSettings_ITP(workspace_dir, data_dir, save_dir)

        f.write('from lib.test.evaluation.environment import EnvSettings\n\n')
        f.write('def local_env_settings():\n')
        f.write('    settings = EnvSettings()\n\n')
        f.write('    # Set your local paths here.\n\n')

        for attr in dir(settings):
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            attr_val = getattr(settings, attr)
            if not attr.startswith('__') and not callable(attr_val):
                if comment_str is None:
                    f.write('    settings.{} = \'{}\'\n'.format(attr, attr_val))
                else:
                    f.write('    settings.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))
        f.write('\n    return settings\n\n')


def env_settings():
    env_module_name = 'lib.test.evaluation.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.local_env_settings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')

        # Create a default file
        create_default_local_file()
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. '
                           'Then try to run again.'.format(env_file))