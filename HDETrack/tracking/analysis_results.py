import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]
import argparse

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--dataset', type=str, default='eventvot', help='coesot or fe108 or visevent or eventvot.')
    parser.add_argument('--parameter_name', type=str, default='hdetrack_eventvot', help='coesot or fe108 or visevent or eventvot.')
    args = parser.parse_args()

    trackers = []
    dataset_name = args.dataset      # coesot  fe108
    """stark"""
    # trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
    #                             run_ids=None, display_name='STARK-S50'))
    # trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
    #                             run_ids=None, display_name='STARK-ST50'))
    # trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
    #                             run_ids=None, display_name='STARK-ST101'))
    """TransT"""
    # trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
    #                             run_ids=None, display_name='TransT_N2', result_only=True))
    # trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
    #                             run_ids=None, display_name='TransT_N4', result_only=True))
    """pytracking"""
    # trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
    # trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
    # trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
    # trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
    # trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
    """HDEtrack"""
    trackers.extend(trackerlist(name='hdetrack', parameter_name=args.parameter_name, dataset_name=dataset_name,
                                run_ids=None, display_name='HDETrack256'))
    dataset = get_dataset(dataset_name)
    # dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
    # plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
    #              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
    print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
    # print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))

