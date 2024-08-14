from pathlib import Path

import argh
import pandas as pd
from argh import arg

from helper_files.metrics import create_popularity_bins, join_interaction_with_country

EXPERIMENTS_FOLDER = Path('experiments')


@arg('experiment', type=str, help='Name of the dataset (a subfolder under data/) to be evaluated')
def analyze_dataset(experiment):
    dataset_path = EXPERIMENTS_FOLDER / experiment / 'input' / 'dataset.inter'
    demographics_path = EXPERIMENTS_FOLDER / experiment / 'input' / 'demographics.tsv'
    tracks_path = EXPERIMENTS_FOLDER / experiment / 'input' / 'tracks.tsv'

    interactions = pd.read_csv(dataset_path, delimiter='\t', header=None, skiprows=1, names=['user_id', 'item_id'])

    tracks_info = pd.read_csv(tracks_path, delimiter='\t', header=None).reset_index()
    tracks_info.columns = ['item_id', 'artist', 'title', 'country']
    demographics = pd.read_csv(demographics_path, delimiter='\t', header=None,
                               names=['country', 'age', 'gender', 'signup_date'])

    tracks_info['country'] = tracks_info['country'].replace('GB', 'UK')
    demographics['country'] = demographics['country'].replace('GB', 'UK')

    tracks_with_popularity = create_popularity_bins(interactions, tracks_info)

    df = join_interaction_with_country(interactions, demographics, tracks_info,
                                                                 tracks_with_popularity)
    total_interactions = len(df)
    total_users = len(df.user_id.unique())
    total_items = len(df.item_id.unique())
    print(f'Number of interactions: {total_interactions}')
    print(f'Number of users: {total_users}')
    print(f'Number of items: {total_items}')
    print(f'Average Interactions per user: {len(df) / len(df.user_id.unique())}')
    print(f'Average Interactions per item: {len(df) / len(df.item_id.unique())}')

    """
    # Latex for table in RecSys paper
    for c in ['US', 'UK', 'DE', 'SE', 'CA', 'FR', 'AU', 'FI', 'NO', 'BR', 'NL', 'PL', 'RU', 'JP', 'IT',]:
        user_count = len(df[df["user_country"] == c]["user_id"].unique())
        item_count = len(df[df["artist_country"] == c]["item_id"].unique())
        user_interactions = len(df[df["user_country"] == c])
        item_interactions = len(df[df["artist_country"] == c])
        print(f'{c} & {item_count:,} & {user_count:,} & {item_interactions:,} & {user_interactions:,} \\\\')
    """

    # Code for generating overall dataset statistics
    for c in ['US', 'UK', 'DE', 'SE', 'CA', 'FR', 'AU', 'FI', 'NO', 'BR', 'NL', 'PL', 'RU', 'JP', 'IT']:
        user_count = len(df[df["user_country"] == c]["user_id"].unique())
        item_count = len(df[df["artist_country"] == c]["item_id"].unique())
        user_interactions = len(df[df["user_country"] == c])
        item_interactions = len(df[df["artist_country"] == c])

        # Item Percentage
        # print(f'{c},{item_count/total_items}')
        # Item Interaction Percentage
        # print(f'{c},{item_interactions/total_interactions}')
        # Average Interactions per item
        # print(f'{c},{item_interactions/item_count}')
        # User Percentage
        # print(f'{c},{user_count/total_users}')
        # User Interaction Percentage
        # print(f'{c},{user_interactions/total_interactions}')
        # Average Interactions per user
        print(f'{c},{user_interactions/user_count}')


if __name__ == '__main__':
    argh.dispatch_command(analyze_dataset)
