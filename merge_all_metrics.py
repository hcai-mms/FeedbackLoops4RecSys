import os

import pandas as pd

experiments_to_evaluate = [
    '100k_sample1_lightgcn_rank_based',
    '100k_sample1_bpr_rank_based',
    '100k_sample1_neumf_rank_based',
    '100k_sample1_itemknn_rank_based',
    '100k_sample1_multivae_rank_based',
    '100k_sample1_pop_rank_based',
]

if __name__ == '__main__':
    metrics = []
    metrics_user_based = []
    for experiment in experiments_to_evaluate:
        if not os.path.exists(os.path.join('experiments', experiment)):
            print(f"ERROR: Skipping {experiment}: Experiment does not exist!")
            continue
        # load metrics from file
        met = pd.read_csv(os.path.join('experiments', experiment, 'metrics.csv'))
        metrics.append(met)
        # load user-based metrics from file
        met_ub = pd.read_csv(os.path.join('experiments', experiment, 'user_based_metrics.csv'))
        metrics_user_based.append(met_ub)

    # merge all metrics into a single DataFrame
    all_metrics = pd.concat(metrics, ignore_index=True)
    # save merged metrics to file
    all_metrics.to_csv('metrics_merged.csv', index=False)

    # merge all user-based metrics into a single DataFrame
    all_metrics_user_based = pd.concat(metrics_user_based, ignore_index=True)
    # save merged user-based metrics to file
    all_metrics_user_based.to_csv('user_based_metrics.csv', index=False)