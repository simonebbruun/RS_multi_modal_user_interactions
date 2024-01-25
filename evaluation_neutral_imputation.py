import pandas as pd
import preprocessing_functions as pf
import evaluation_functions as ef
from tensorflow.keras.models import load_model


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
conversations = pd.concat((pd.read_csv('data_conversations_embedding_1.csv'),pd.read_csv('data_conversations_embedding_2.csv')),axis=0)
sessions = pd.read_csv('data_sessions.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting train events.
users = pf.train_test_split(users, test = True)

conv_ses = pf.event_type(users)

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]

events = users['purchase_event_id'].drop_duplicates().sort_values().reset_index(drop=True)


# Selecting conversations.
users_conv = users[~users['conversation_id'].isna()]

# Averaging conversations.
users_conv = users_conv[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')
users_conv = users_conv.drop(['sentence_number', 'sentence_speaker'], axis=1)

users_conv = users_conv.drop(['event_number', 'conversation_id'], axis=1)
users_conv = users_conv.groupby(['purchase_event_id'], as_index=False).mean()

# Imputing conversations.
users_conv = users[['purchase_event_id']].drop_duplicates().merge(users_conv, how = 'left')

users_conv = pf.imputing_conversations(users_conv, test = True)

# Convert conversations from dataframe to numpy.
users_conv = users_conv.sort_values(by=['purchase_event_id']).reset_index(drop = True)
users_conv = users_conv.drop(['purchase_event_id'], axis = 1).values


# Selecting sessions.
users_ses = users[~users['session_id'].isna()]
users_ses = users_ses[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')

# Removing low frequent action_tags.
users_ses = pf.remove_low_frequent_features(users_ses, features='action_tags', id_columns=['purchase_event_id', 'session_id'], number_column='action_number', model_name='imputation', test = True)

# Encoding sessions.
users_ses = pf.encoding(users_ses, features='action_tags', id_columns=['session_id'], number_column='action_number', model_name='imputation', test = True)

# Imputing sessions.
users_ses = users[['purchase_event_id']].drop_duplicates().merge(users_ses, how = 'left')
users_ses = pf.imputing_sessions(users_ses, test = True)

# Padding sessions.
users_ses = pf.padding(users_ses, ['session_id'])


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
evaluation_list_union = []
for i in range(1,6):
    model = load_model('model_imputation_'+str(i)+'.h5')

    predictions = model.predict([users_conv, users_ses])*post_filter
    
    hit = ef.hit(predictions, purchases, k)
    average_precision = ef.average_precision(predictions, purchases, k)
    
    evaluation_avg = pd.DataFrame({'purchase_event_id' : events, 'hit_rate' : hit, 'mean_average_precision' : average_precision})
    evaluation_avg = conv_ses.merge(evaluation_avg, how = 'left')
    evaluation_list.append(evaluation_avg[['event', 'hit_rate', 'mean_average_precision']].groupby('event').mean())
    evaluation_list_union.append(evaluation_avg[['hit_rate', 'mean_average_precision']].mean())
evaluation_avg = pd.concat(evaluation_list).groupby('event').mean()
evaluation_avg_union = pd.concat(evaluation_list_union, axis = 1).mean(axis=1)


# Varying thresholds.
evaluation_thresholds_list = []
for i in range(1,6):
    model = load_model('model_imputation_'+str(i)+'.h5')

    predictions = model.predict([users_conv, users_ses])*post_filter
    
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
