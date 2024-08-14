import json
from pathlib import Path

import argh
import pandas as pd
from argh import arg

from helper_files.metrics import create_popularity_bins, join_interaction_with_country

EXPERIMENTS_FOLDER = Path('experiments')

def normalize_text(text: str) -> str:
    text = str(text).lower()
    pre_special_chars = text
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = text.replace(',', '')
    text = text.replace(';', '')
    text = text.replace('!', '')
    text = text.replace('?', '')
    text = text.replace('.', '')
    text = text.replace(':', '')
    text = text.replace('-', ' ')
    text = text.replace('(', '')
    text = text.replace(')', '')
    text = text.replace('[', '')
    text = text.replace(']', '')
    text = text.replace('{', '')
    text = text.replace('}', '')
    text = text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace('`', '')
    text = text.replace('´', '')
    text = text.replace('’', '')
    return text

@arg('experiment', type=str, help='Name of the dataset (a subfolder under data/) to be evaluated')
def analyze_dataset(experiment):
    tracks_path = EXPERIMENTS_FOLDER / experiment / 'input' / 'tracks.tsv'
    tracks = pd.read_csv(tracks_path, delimiter='\t', header=None).reset_index()
    tracks.columns = ['item_id', 'artist', 'title', 'country']
    tracks['country'] = tracks['country'].replace('GB', 'UK')
    tracks['artist'] = tracks['artist'].apply(normalize_text)
    tracks['title'] = tracks['title'].apply(normalize_text)

    m4a_tracks = pd.read_csv('m4a_tracks.tsv', delimiter='\t')
    m4a_lang = pd.read_csv('m4a_lang.tsv', delimiter='\t')

    # merge m4a based on id column
    m4a = pd.merge(m4a_tracks, m4a_lang, on='id')
    m4a['song'] = m4a['song'].apply(normalize_text)
    m4a['artist'] = m4a['artist'].apply(normalize_text)

    # merge lang column of m4a into tracks based on track name and artist
    tracks = pd.merge(tracks, m4a, how='left', left_on=['title', 'artist'], right_on=['song', 'artist'])
    # fill nan with 'n/a'
    tracks['lang'] = tracks['lang'].fillna('n/a')
    print('Languages distribution after first merge')
    print(tracks.value_counts('lang'))

    langs = [
        'en', 'pt', 'es', 'fr', 'ko', 'de', 'pl', 'ja', 'it', 'sv', 'id', 'no', 'ru', 'fi',
        'tr', 'nl', 'so', 'hr', 'uk', 'da', 'ca', 'et', 'tl', 'ro', 'af', 'sw', 'hu', 'cy', 'INTRUMENTAL', 'n/a'
    ]

    for c in ['US', 'UK', 'DE', 'SE', 'CA', 'FR', 'AU', 'FI', 'NO', 'BR', 'NL', 'PL', 'RU', 'JP', 'IT']:
        counts = tracks[tracks['country'] == c].value_counts('lang')
        # fill missing languages with 0
        counts = counts.reindex(langs, fill_value=0)
        print(f'{c} languages distribution')
        for val in counts:
            print(val)


if __name__ == '__main__':
    argh.dispatch_command(analyze_dataset)
