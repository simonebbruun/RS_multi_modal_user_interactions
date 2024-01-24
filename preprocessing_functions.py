import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from pickle import load
from pickle import dump


def train_test_split(dataframe, test = False):
    purchase_events = round(dataframe['purchase_event_id'].nunique()*0.9)
    if test:
        dataframe = dataframe[dataframe['purchase_event_id']>=purchase_events]
    else:
        dataframe = dataframe[dataframe['purchase_event_id']<purchase_events]
    
    return dataframe


def remove_low_frequent_features(dataframe, features, id_columns, number_column, model_name, test = False):
    dataframe = dataframe.join(dataframe[features].str.split(',', expand=True).stack().reset_index(level=1, drop=True).rename('feature'))
    if test:
        features_pct = pd.read_csv(features+'_'+model_name+'.csv', index_col = 0)
    else:
        features_total = len(dataframe[[*id_columns, number_column]].drop_duplicates())
        features_pct = dataframe['feature'].value_counts()/features_total
        features_pct.to_csv(features+'_'+model_name+'.csv')
    
    dataframe = dataframe.merge(features_pct, how = 'left', left_on='feature', right_index=True, suffixes=(None, '_pct'))
    dataframe_features = dataframe[dataframe['feature_pct']>=0.001]
    dataframe = dataframe[id_columns].drop_duplicates()
    dataframe = dataframe.merge(dataframe_features, how = 'left')
    dataframe_features = dataframe[~dataframe['feature'].isna()]
    dataframe = dataframe[['purchase_event_id']].drop_duplicates().merge(dataframe_features, how = 'left')
    dataframe['feature'] = dataframe['feature'].fillna('')
    grp_columns = [c for c in dataframe.columns if c not in ['user_id', features, 'feature', 'feature_pct']]
    dataframe = dataframe.groupby(grp_columns, dropna=False, as_index=False)['feature'].apply(','.join).rename(columns={'feature': features})
    
    return dataframe


def encoding(dataframe, features, id_columns, number_column, model_name, test = False):
    # Multi-hot encoding.
    if test:
        multi_hot_encoder_session = load(open('multi_hot_encoder_'+model_name+'.pkl', 'rb'))

        dummies = multi_hot_encoder_session.fit_transform(dataframe[features].str.split(','))
        dummy_names = multi_hot_encoder_session.classes_
        dataframe = pd.concat([dataframe, pd.DataFrame(dummies, columns = dummy_names)], axis=1)
        dataframe = dataframe.drop([features], axis=1)
    else:
        multi_hot_encoder_session = MultiLabelBinarizer()
        dummies = multi_hot_encoder_session.fit_transform(dataframe[features].str.split(','))
        dummy_names = multi_hot_encoder_session.classes_
        dataframe = pd.concat([dataframe, pd.DataFrame(dummies, columns = dummy_names)], axis=1)
        dataframe = dataframe.drop([features], axis=1)
    
        multi_hot_encoder_session = MultiLabelBinarizer(classes=dummy_names)
        dump(multi_hot_encoder_session, open('multi_hot_encoder_'+model_name+'.pkl', 'wb'))
    
    # Max-pooling operation.
    dataframe = dataframe.groupby(['purchase_event_id', 'event_number', *id_columns], dropna = False, as_index = False).max()
    dataframe = dataframe.drop([number_column], axis = 1)
    
    return dataframe


def padding(dataframe, id_columns, shuffle = False):
    dataframe = dataframe.sort_values(by=['purchase_event_id', 'event_number']).reset_index(drop=True)
    dataframe = dataframe.drop(['event_number', *id_columns], axis=1)

    purchase_event_id = dataframe.columns.get_loc('purchase_event_id')

    data_array = np.array(list(dataframe.groupby(['purchase_event_id'], dropna = False).apply(pd.DataFrame.to_numpy)))
    if shuffle:
        for i in range(len(data_array)):
            np.random.shuffle(data_array[i])
    
    n_events = len(data_array)
    n_conv_ses = max([len(data_array[i]) for i in range(n_events)])
    n_features = len(dataframe.columns)

    data_padded = np.empty((n_events,n_conv_ses,n_features))
    for i in range(0,n_events):
        data_padded[i] = np.pad(data_array[i], ((n_conv_ses-len(data_array[i]),0), (0, 0)), 'constant')
        
    data_padded = np.delete(data_padded, [purchase_event_id], 2)
    
    return data_padded


def imputing_conversations(dataframe, test = False):
    embedding_columns = [c for c in dataframe.columns if c not in ['purchase_event_id']]
    if test:
        imputer_conversation = load(open('imputer_conversations.pkl', 'rb'))
        dataframe[embedding_columns] = imputer_conversation.transform(dataframe[embedding_columns])
    else:
        imputer_conversation = SimpleImputer()
        dataframe[embedding_columns] = imputer_conversation.fit_transform(dataframe[embedding_columns])
        dump(imputer_conversation, open('imputer_conversations.pkl', 'wb'))
    
    return dataframe


def imputing_sessions(dataframe, test = False):
    action_columns = [c for c in dataframe.columns if c not in ['purchase_event_id', 'event_number', 'session_id']]
    if test:
        imputer_conversation = load(open('imputer_sessions.pkl', 'rb'))
        dataframe[action_columns] = imputer_conversation.transform(dataframe[action_columns])
    else:
        imputer_conversation = SimpleImputer(strategy = 'most_frequent')
        dataframe[action_columns] = imputer_conversation.fit_transform(dataframe[action_columns])
        dump(imputer_conversation, open('imputer_sessions.pkl', 'wb'))
    
    dataframe['event_number'] = dataframe['event_number'].fillna(1)
    
    return dataframe


def binarize_items(dataframe):
    multi_label_binarizer = MultiLabelBinarizer()
    dataframe = pd.concat([dataframe.reset_index(drop=True), pd.DataFrame(multi_label_binarizer.fit_transform(dataframe['item_id'].astype(str).str.split(';')), columns=multi_label_binarizer.classes_)], axis=1)
    dataframe = dataframe.drop(['item_id'], axis=1)
    dataframe = dataframe.groupby(['purchase_event_id'], as_index=False).max()

    return dataframe


def event_type(dataframe):
    dataframe_copy = dataframe.copy(deep=True)
    dataframe_copy['conversation_indicator'] = np.where(dataframe_copy['session_id'].isna(), 1, 0)
    dataframe_copy['session_indicator'] = np.where(dataframe_copy['session_id'].isna(), 0, 1)

    dataframe_copy = dataframe_copy.groupby('purchase_event_id', as_index=False)[['conversation_indicator', 'session_indicator']].max()
    dataframe_copy['event'] = np.where((dataframe_copy['conversation_indicator']==1) & (dataframe_copy['session_indicator']==0),'conversation',
                                 np.where((dataframe_copy['conversation_indicator']==0) & (dataframe_copy['session_indicator']==1),'session',
                                          'intersection'))
    dataframe_copy = dataframe_copy.drop(['conversation_indicator', 'session_indicator'], axis = 1)
    
    return dataframe_copy


def pipeline_session(dataframe):
    # Removing low frequent action_tags.
    dataframe = remove_low_frequent_features(dataframe, 'action_tags', ['purchase_event_id', 'session_id'], 'action_number', 'session', test = True)

    # Encoding sessions.
    dataframe = encoding(dataframe, features='action_tags',  id_columns=['session_id'], number_column='action_number', model_name='session', test = True)

    # Padding sessions.
    dataframe = padding(dataframe, ['session_id'])
       
    return dataframe


def pipeline_conversation(dataframe):
    # Averaging embeddings.
    dataframe = dataframe.drop(['event_number', 'conversation_id', 'sentence_number', 'sentence_speaker'], axis=1)
    dataframe = dataframe.groupby('purchase_event_id').mean()
    dataframe = dataframe.values
    
    return dataframe