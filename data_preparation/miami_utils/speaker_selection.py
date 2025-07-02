

def select_speakers(data, speakers):
    speakers = speakers.drop_duplicates()

    # get statistics about number of utterances per speaker etc.
    stats = data.groupby('spk').count()[['utt']].copy()
    stats.columns = ['total']
    stats['eng'] = data[(data['eng']) & (~data['spa'])].groupby('spk').count()['utt']
    stats['spa'] = data[(data['spa']) & (~data['eng'])].groupby('spk').count()['utt']
    stats['cs'] = data[data['codeswitching']].groupby('spk').count()['utt']
    stats = stats.convert_dtypes()
    stats = stats.merge(speakers[['name', 'gender', 'age']], right_on='name', left_index=True, how='left').set_index('name')

    # select only speakers with at least 20 utterances in each language setting
    sub = stats[(stats['eng'] > 19) & (stats['spa'] > 19) & (stats['cs'] > 19)]

    print(f'Total number of speakers: {len(sub)}')  # 25 total speakers
    print(f'Number of female speakers: {len(sub[sub["gender"] == "female"])}')  # 17 female speakers
    print(f'Number of male speakers: {len(sub[sub["gender"] == "male"])}')   # 8 male speakers

    # segment data based on selected speakers
    selected_speakers = sub['gender'].to_dict()
    sub_data = data[data['spk'].isin(list(selected_speakers.keys()))].copy()
    sub_data['gender'] = sub_data['spk'].apply(lambda x: selected_speakers[x])
    return sub_data, selected_speakers