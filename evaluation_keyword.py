import pandas as pd
import preprocessing_functions as pf
import numpy as np
import evaluation_functions as ef
from tensorflow.keras.models import load_model


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
conversations = pd.read_csv('data_conversations_keyword.csv')
sessions = pd.read_csv('data_sessions.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting test events.
users = pf.train_test_split(users, test = True)

conv_ses = pf.event_type(users)

users_conversation = users[~users['conversation_id'].isna()].merge(conversations, how = 'left')
users_conversation = users_conversation.drop(['sentence_speaker'], axis=1)
users_conversation = users_conversation.rename(columns = {'sentence_number' : 'number'})
users_session = users[~users['session_id'].isna()].merge(sessions, how = 'left')
users_session = users_session.rename(columns = {'action_number' : 'number', 'action_tags' : 'keywords'})
users = pd.concat([users_conversation, users_session], axis=0)
users_keyword = users[~users['keywords'].isna()]
users_keyword = users_keyword[users_keyword['keywords']!='']
users = users[['purchase_event_id']].drop_duplicates().merge(users_keyword, how = 'left')

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]

events = users['purchase_event_id'].drop_duplicates().sort_values().reset_index(drop=True)


# Removing low frequent keywords.
users = pf.remove_low_frequent_features(users, features='keywords', id_columns=['purchase_event_id', 'conversation_id', 'session_id'], number_column='number', model_name='keyword', test = True)

# Encoding conversations and sessions.
users['conversation_ind'] = np.where(users['session_id'].isna(), 1, 0)
users = pf.encoding(users, features='keywords', id_columns=['conversation_id', 'session_id'], number_column='number', model_name='keyword', test = True)


# Binarizing purchase events.
purchases = pf.binarize_items(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


# Padding conversations and sessions.
users_padded = pf.padding(users, ['conversation_id', 'session_id'])


''' Evaluation. '''
# Average of 5 runs.
k = 3
    
evaluation_list = []
evaluation_list_union = []
for i in range(1,6):
    model = load_model('model_keyword_'+str(i)+'.h5')

    predictions = model.predict(users_padded)*post_filter
    
    hit = ef.hit(predictions, purchases, k)
    average_precision = ef.average_precision(predictions, purchases, k)
    
    evaluation_avg = pd.DataFrame({'purchase_event_id' : events, 'hit_rate' : hit, 'mean_average_precision' : average_precision})
    evaluation_avg = conv_ses.merge(evaluation_avg, how = 'left')
    evaluation_list.append(evaluation_avg[['event', 'hit_rate', 'mean_average_precision']].groupby('event', as_index = False).mean())
    evaluation_list_union.append(evaluation_avg[['hit_rate', 'mean_average_precision']].mean())
evaluation_avg = pd.concat(evaluation_list).groupby('event', as_index = False).mean()
evaluation_avg_union = pd.DataFrame(pd.concat(evaluation_list_union, axis = 1).mean(axis=1)).transpose()
evaluation_avg_union['event'] = 'union'
evaluation_avg = pd.concat((evaluation_avg, evaluation_avg_union))


# Varying thresholds.
evaluation_thresholds_list = []
for i in range(1,6):
    model = load_model('model_keyword_'+str(i)+'.h5')

    predictions = model.predict(users_padded)*post_filter
    
    evaluation_thresholds = []
    evaluation_thresholds_union = []
    for k in range(1,6):
        
        average_precision = ef.average_precision(predictions, purchases, k)
        
        evaluation_threshold = pd.DataFrame({'purchase_event_id' : events, str(k) : average_precision})
        evaluation_threshold = conv_ses.merge(evaluation_threshold, how = 'left')
        evaluation_thresholds.append(evaluation_threshold[['event', str(k)]].groupby('event').mean())
        evaluation_thresholds_union.append(evaluation_threshold[[str(k)]].mean())
    evaluation_thresholds = pd.concat(evaluation_thresholds, axis = 1).transpose()
    evaluation_thresholds['union'] = pd.concat(evaluation_thresholds_union)
    evaluation_thresholds_list.append(evaluation_thresholds)
evaluation_thresholds = pd.concat(evaluation_thresholds_list)
evaluation_thresholds = evaluation_thresholds.groupby(evaluation_thresholds.index).mean()