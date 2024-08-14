import os.path

from helper_files.data_loader import load_data

experiments_to_evaluate = [
    'example',
    # add more experiments here
]

if __name__ == '__main__':
    for experiment in experiments_to_evaluate:
        if not os.path.exists(os.path.join('experiments', experiment)):
            print(f"ERROR: Skipping {experiment}: Experiment does not exist!")
            continue
        print(f'Processing experiment "{experiment}"')
        experiment_folder = os.path.join('experiments', experiment)

        # delete previous metrics file
        if os.path.exists(os.path.join(experiment_folder, 'metrics.csv')):
            os.remove(os.path.join(experiment_folder, 'metrics.csv'))
        if os.path.exists(os.path.join(experiment_folder, 'user_based_metrics.csv')):
            os.remove(os.path.join(experiment_folder, 'user_based_metrics.csv'))
        if os.path.exists(os.path.join(experiment_folder, 'baselines.csv')):
            os.remove(os.path.join(experiment_folder, 'baselines.csv'))
        load_data(experiments_folder='experiments', experiment_name=experiment)
