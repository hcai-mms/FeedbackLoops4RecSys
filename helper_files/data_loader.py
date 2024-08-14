import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from helper_files.metrics import calculate_baselines, calculate_proportions, join_interaction_with_country, \
    calculate_iteration_jsd_per_user, calculate_user_bin_jsd, create_popularity_bins, merge_jsd_dataframes, calculate_country_distribution


def load_iteration_data(experiment_dir, iteration_number, top_k=True):
    """
    Load the top-k recommendations for a given iteration.

    Parameters:
    - experiment_dir: The directory containing the experiment data.
    - iteration_number: The iteration number to load.

    Returns:
    DataFrame containing the top-k recommendations for the given iteration.
    """
    if top_k:
        data_file = f'output/iteration_{iteration_number}_top_k.tsv'
        use_cols = ['user_id', 'item_id']
    else:
        data_file = f'datasets/iteration_{iteration_number}.inter'
        use_cols = ['user_id:token', 'item_id:token']
    data_path = os.path.join(experiment_dir, data_file)

    data = pd.read_csv(data_path, delimiter='\t', usecols=use_cols)
    if not top_k:
        data.rename(columns={'user_id:token': 'user_id', 'item_id:token': 'item_id'}, inplace=True)
    return data


def load_data(experiments_folder, experiment_name, focus_country='US', control_country="DE"):
    """Loads the data and metrics of an experiment. If the metrics dont exist yet, they are computed on-demand"""
    input_dir_path = os.path.join(experiments_folder, experiment_name, 'input')
    params_path = os.path.join(experiments_folder, experiment_name, 'params.json')
    demographics_file = os.path.join(input_dir_path, 'demographics.tsv')
    dataset_inter_filepath = os.path.join(input_dir_path, 'dataset.inter')
    tracks_filepath = os.path.join(input_dir_path, 'tracks.tsv')
    iterations = 0

    with open(params_path) as f:
        params_dict = json.load(f)

    global_interactions = pd.read_csv(dataset_inter_filepath, delimiter='\t', header=None, skiprows=1, names=['user_id', 'item_id'])
    tracks_info = pd.read_csv(tracks_filepath, delimiter='\t', header=None).reset_index()
    tracks_info.columns = ['item_id', 'artist', 'title', 'country']
    demographics = pd.read_csv(demographics_file, delimiter='\t', header=None, names=['country', 'age', 'gender', 'signup_date'])

    tracks_info['country'] = tracks_info['country'].replace('GB', 'UK')
    demographics['country'] = demographics['country'].replace('GB', 'UK')

    # If baselines.csv does exist, load it, else calculate it and save the data.
    if os.path.exists(os.path.join(experiments_folder, experiment_name, 'baselines.csv')):
        print('Loading Baselines from CSV File...')
        baselines = pd.read_csv(os.path.join(experiments_folder, experiment_name, 'baselines.csv'))
    else:
        print('Calculating Baselines...')
        baselines = calculate_baselines(global_interactions, tracks_info, demographics, focus_country)
        # Saving the results to CSV
        baselines.to_csv(os.path.join(experiments_folder, experiment_name, 'baselines.csv'), index=False)

    # If csv does exist, load it, else calculate it and save the data.
    if os.path.exists(os.path.join(experiments_folder, experiment_name, 'metrics.csv')) and os.path.exists(os.path.join(experiments_folder, experiment_name, 'user_based_metrics.csv')):
        print('Loading Metrics from CSV Files...')
        metric_data = pd.read_csv(os.path.join(experiments_folder, experiment_name, 'metrics.csv'))
        user_based_metric_data = pd.read_csv(os.path.join(experiments_folder, experiment_name, 'user_based_metrics.csv'))
    else:
        # Load global interactions and tracks info
        tracks_info = pd.read_csv(tracks_filepath, delimiter='\t', header=None).reset_index()
        tracks_info.columns = ['item_id', 'artist', 'title', 'country']
        demographics = pd.read_csv(demographics_file, delimiter='\t', header=None, names=['country', 'age', 'gender', 'signup_date'])

        tracks_info['country'] = tracks_info['country'].replace('GB', 'UK')
        demographics['country'] = demographics['country'].replace('GB', 'UK')

        tracks_with_popularity = create_popularity_bins(global_interactions, tracks_info)

        original_interactions_merged = join_interaction_with_country(global_interactions, demographics, tracks_info, tracks_with_popularity)

        print(f'Total items: {len(original_interactions_merged["item_id"].unique())}')
        print(f'Total users: {len(original_interactions_merged["user_id"].unique())}')
        print(f'Total interactions: {len(original_interactions_merged)}')

        # Calculate the number of iterations
        iterations = len(os.listdir(os.path.join(experiments_folder, experiment_name, 'datasets')))

        metric_data, user_based_metric_data = calculate_prop_jsd(experiments_folder, experiment_name, iterations, tracks_info, demographics,
                                                                 params_dict, original_interactions_merged, tracks_with_popularity, focus_country)

    # Dictionary to hold dataframes separated by model and choice_model
    model_data_dict = {}
    unique_combinations = metric_data[['model', 'choice_model']].drop_duplicates()

    for _, row in unique_combinations.iterrows():
        model = row['model']
        choice_model = row['choice_model']

        # Filter the dataframe for each combination of model and choice_model
        filtered_df = metric_data[(metric_data['model'] == model) & (metric_data['choice_model'] == choice_model)]

        # Apply naming conventions
        if choice_model == 'rank_based':
            choice_model = 'Rank Based'
        elif choice_model == 'us_centric':
            choice_model = 'US Centric'

        model_config = (model, choice_model)
        model_data_dict[model_config] = {
            'baselines': baselines,
            'params_dict': params_dict,
            'metrics': filtered_df,
            # 'inter_proportions': filtered_df[filtered_df['country'] == control_country][f'interaction_{focus_country.lower()}_proportion'].tolist(),
            # 'jsd_values': filtered_df[filtered_df['country'] == control_country]['jsd'].tolist(),
            # 'inter_jsd_value': filtered_df[filtered_df['country'] == control_country]['interaction_jsd'].tolist()
        }

    user_model_data_dict = {}
    unique_combinations = user_based_metric_data[['model', 'choice_model']].drop_duplicates()

    for _, row in unique_combinations.iterrows():
        model = row['model']
        choice_model = row['choice_model']

        # Filter the dataframe for each combination of model and choice_model
        filtered_df = user_based_metric_data[(user_based_metric_data['model'] == model) & (user_based_metric_data['choice_model'] == choice_model)]

        # Apply naming conventions
        if choice_model == 'rank_based':
            choice_model = 'Rank Based'
        elif choice_model == 'us_centric':
            choice_model = 'US Centric'

        model_config = (model, choice_model)
        user_model_data_dict[model_config] = {
            'baselines': baselines,
            'params_dict': params_dict,
            'metrics': filtered_df,
            # 'inter_proportions': filtered_df[filtered_df['country'] == control_country][f'interaction_{focus_country.lower()}_proportion'].tolist(),
            # 'jsd_values': filtered_df[filtered_df['country'] == control_country]['jsd'].tolist(),
            # 'inter_jsd_value': filtered_df[filtered_df['country'] == control_country]['interaction_jsd'].tolist()
        }

    return model_data_dict, user_model_data_dict


def calculate_prop_jsd(experiments_folder, experiment_name, iterations, tracks_info, demographics, params_dict, original_interactions_merged, tracks_with_popularity, focus_country):
    """Calculates US/Local/Other proportions and JSD divergence scores"""
    column_dtypes = {
        'user_id': 'int',
        'model': 'str',
        'choice_model': 'str',
        'iteration': 'int',
        'country': 'str',
        'user_count': 'int',
        'jsd': 'float',
        'bin_jsd': 'float',
        f'{focus_country.lower()}_proportion': 'float',
        'local_proportion': 'float',
        f'interaction_{focus_country.lower()}_proportion': 'float',
        'interaction_local_proportion': 'float',
        'interaction_jsd': 'float',
        'interaction_bin_jsd': 'float'
    }

    metric_data = pd.DataFrame(columns=column_dtypes.keys()).astype(column_dtypes)
    user_based_metric_data = pd.DataFrame(columns=column_dtypes.keys()).astype(column_dtypes)

    unique_item_countries = tracks_info['country'].unique()
    user_ids = original_interactions_merged['user_id'].unique()
    interactions_by_user = original_interactions_merged.groupby('user_id')
    history_distribution = np.zeros((len(user_ids), len(unique_item_countries)))

    for user_id in range(0, len(user_ids)):
        user_interactions = interactions_by_user.get_group(user_id)
        history_distribution[user_id] = calculate_country_distribution(user_interactions, unique_item_countries)

    print('Calculating JSD values and recommendation proportions. This may take a while...')
    for iteration in tqdm(range(1, iterations), desc='Processing Iterations'):
        top_k_data = load_iteration_data(os.path.join(experiments_folder, experiment_name), iteration, True)
        user_based_proportion_df, proportion_df = calculate_proportions(top_k_data, tracks_info, demographics, focus_country)
        recs_merged = join_interaction_with_country(top_k_data, demographics, tracks_info, tracks_with_popularity)

        user_based_jsd_df, jsd_df = calculate_iteration_jsd_per_user(recs_merged, unique_item_countries, history_distribution, params_dict["model"], params_dict["choice_model"], iteration, user_ids, focus_country)
        user_based_bin_jsd_df = calculate_user_bin_jsd(recs_merged, original_interactions_merged)
        user_based_jsd_df, jsd_df = merge_jsd_dataframes(jsd_df, user_based_jsd_df, user_based_bin_jsd_df)

        jsd_df[f'{focus_country.lower()}_proportion'] = proportion_df[f'{focus_country.lower()}_proportion']
        jsd_df['local_proportion'] = proportion_df['local_proportion']

        user_based_jsd_df[f'{focus_country.lower()}_proportion'] = user_based_proportion_df[f'{focus_country.lower()}_proportion']
        user_based_jsd_df['local_proportion'] = user_based_proportion_df['local_proportion']

        # Calculate input proportions
        current_iteration_input = load_iteration_data(os.path.join(experiments_folder, experiment_name), iteration, False)

        user_based_input_proportion_df, input_proportion_df = calculate_proportions(current_iteration_input, tracks_info, demographics, focus_country)
        input_merged = join_interaction_with_country(current_iteration_input, demographics, tracks_info, tracks_with_popularity)

        jsd_df[f'interaction_{focus_country.lower()}_proportion'] = input_proportion_df[f'{focus_country.lower()}_proportion']
        jsd_df['interaction_local_proportion'] = input_proportion_df['local_proportion']

        user_based_jsd_df[f'interaction_{focus_country.lower()}_proportion'] = user_based_input_proportion_df[f'{focus_country.lower()}_proportion']
        user_based_jsd_df['interaction_local_proportion'] = user_based_input_proportion_df['local_proportion']

        # Calculate input JSD
        user_based_input_jsd_df, input_jsd_df = calculate_iteration_jsd_per_user(input_merged, unique_item_countries, history_distribution,
                                                                                 params_dict["model"], params_dict["choice_model"], iteration, user_ids, focus_country)
        user_based_input_bin_jsd_df = calculate_user_bin_jsd(input_merged, original_interactions_merged)
        user_based_input_jsd_df, input_jsd_df = merge_jsd_dataframes(input_jsd_df, user_based_input_jsd_df, user_based_input_bin_jsd_df)

        input_jsd_df = input_jsd_df[['jsd', 'jsd_summarized', 'bin_jsd']].rename(columns={
            'jsd': 'interaction_jsd',
            'jsd_summarized': 'interaction_jsd_summarized',
            'bin_jsd': 'interaction_bin_jsd'
        })
        user_based_input_jsd_df = user_based_input_jsd_df[['jsd', 'jsd_summarized', 'bin_jsd']].rename(columns={
            'jsd': 'interaction_jsd',
            'jsd_summarized': 'interaction_jsd_summarized',
            'bin_jsd': 'interaction_bin_jsd'
        })

        jsd_df = pd.concat([jsd_df, input_jsd_df], axis=1)
        metric_data = pd.concat([metric_data, jsd_df], ignore_index=True)

        user_based_jsd_df = pd.concat([user_based_jsd_df, user_based_input_jsd_df], axis=1)
        user_based_metric_data = pd.concat([user_based_metric_data, user_based_jsd_df], ignore_index=True)

    # Save metrics to CSV
    csv_save_path = os.path.join(experiments_folder, experiment_name, 'metrics.csv')
    metric_data.to_csv(csv_save_path, index=False)

    csv_save_path = os.path.join(experiments_folder, experiment_name, 'user_based_metrics.csv')
    user_based_metric_data.to_csv(csv_save_path, index=False)

    print("Loaded data successfully")

    return metric_data, user_based_metric_data
