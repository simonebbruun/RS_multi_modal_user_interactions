import pandas as pd
import preprocessing_functions as pf
from tensorflow.keras.models import load_model
import evaluation_functions as ef


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
sessions = pd.read_csv('data_sessions.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting test events.
users = pf.train_test_split(users, test = True)

conv_ses = pf.event_type(users)

users = users[~users['session_id'].isna()]
users = users[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]

events = users['purchase_event_id'].drop_duplicates().sort_values().reset_index(drop=True)


# Removing low frequent action_tags.
users = pf.remove_low_frequent_features(users, 'action_tags', ['purchase_event_id', 'session_id'], 'action_number', 'session', test = True)

# Encoding sessions.
users = pf.encoding(users, features='action_tags',  id_columns=['session_id'], number_column='action_number', model_name='session', test = True)

# Padding sessions.
users_padded = pf.padding(users, ['session_id'])


# Binarizing purchase events.
purchases = pf.binarize_items(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


''' Evaluation. '''
# Average of 5 runs.
k = 3
    
evaluation_list = []
for i in range(1,6):
    model = load_model('model_session_'+str(i)+'.h5')

    predictions = model.predict(users_padded)*post_filter
    
    hit = ef.hit(predictions, purchases, k)
    average_precision = ef.average_precision(predictions, purchases, k)
    
    evaluation_avg = pd.DataFrame({'purchase_event_id' : events, 'hit_rate' : hit, 'mean_average_precision' : average_precision})
    evaluation_avg = conv_ses.merge(evaluation_avg, how = 'left')
    evaluation_list.append(evaluation_avg[['event', 'hit_rate', 'mean_average_precision']].groupby('event', as_index = False).mean())
evaluation_avg = pd.concat(evaluation_list).groupby('event', as_index = False).mean()


# Varying thresholds.
evaluation_thresholds_list = []
for i in range(1,6):
    model = load_model('model_session_'+str(i)+'.h5')

    predictions = model.predict(users_padded)*post_filter
    
    evaluation_thresholds = []
    for k in range(1,6):
        
        average_precision = ef.average_precision(predictions, purchases, k)
        
        evaluation_threshold = pd.DataFrame({'purchase_event_id' : events, str(k) : average_precision})
        evaluation_threshold = conv_ses.merge(evaluation_threshold, how = 'left')
        evaluation_thresholds.append(evaluation_threshold[['event', str(k)]].groupby('event').mean())
    evaluation_thresholds_list.append(pd.concat(evaluation_thresholds, axis = 1).transpose())

evaluation_thresholds = pd.concat(evaluation_thresholds_list)
evaluation_thresholds = evaluation_thresholds.groupby(evaluation_thresholds.index).mean()