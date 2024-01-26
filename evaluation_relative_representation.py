import pandas as pd
import preprocessing_functions as pf
from pickle import load
from sklearn.metrics.pairwise import cosine_similarity
import evaluation_functions as ef
from tensorflow.keras.models import load_model


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
conversations = pd.concat((pd.read_csv('data_conversations_embedding/data_conversations_embedding_1.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_2.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_3.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_4.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_5.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_6.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_7.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_8.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_9.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_10.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_11.csv'),
                           pd.read_csv('data_conversations_embedding/data_conversations_embedding_12.csv')),axis=0)
sessions = pd.read_csv('data_sessions.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting test events.
users = pf.train_test_split(users, test = True)

conv_ses = pf.event_type(users)

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]


# Selecting conversations.
users_conv = users[~users['conversation_id'].isna()]

# Averaging conversations.
users_conv = users_conv[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')
users_conv = users_conv.drop(['sentence_number', 'sentence_speaker'], axis=1)

users_conv = users_conv.groupby(['purchase_event_id', 'event_number', 'conversation_id'], as_index=False).mean()
users_conv['conversation_ind'] = 1

# Computing relative representations of conversations.
encoder = load_model('encoder_conversation.h5')
encodings = encoder.predict(users_conv.drop(['purchase_event_id','event_number','conversation_id', 'conversation_ind'], axis=1))

anchors = load(open('anchors_conversation.npy', 'rb'))

relative_representations = cosine_similarity(encodings,anchors)

users_conv = pd.concat([users_conv[['purchase_event_id','event_number','conversation_id', 'conversation_ind']], pd.DataFrame(relative_representations, columns = ['rel_' + str(i + 1) for i in range(relative_representations.shape[1])])], axis=1)


# Selecting sessions.
users_ses = users[~users['session_id'].isna()]
users_ses = users_ses[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')

# Removing low frequent action_tags.
users_ses = pf.remove_low_frequent_features(users_ses, features='action_tags', id_columns=['purchase_event_id', 'session_id'], number_column='action_number', model_name='relative_representation', test = True)

# Encoding sessions.
users_ses = pf.encoding(users_ses, features='action_tags', id_columns=['session_id'], number_column='action_number', model_name='relative_representation', test = True)
users_ses['conversation_ind'] = 0

# Computing relative representations of sessions.
encoder = load_model('encoder_session.h5')
encodings = encoder.predict(users_ses.drop(['purchase_event_id','event_number','session_id', 'conversation_ind'], axis=1))

anchors = load(open('anchors_session.npy', 'rb'))

relative_representations = cosine_similarity(encodings,anchors)

users_ses = pd.concat([users_ses[['purchase_event_id','event_number','session_id', 'conversation_ind']], pd.DataFrame(relative_representations, columns = ['rel_' + str(i + 1) for i in range(relative_representations.shape[1])])], axis=1)


# Joining conversations and sessions.
users = pd.concat([users_conv, users_ses], axis = 0)

events = users['purchase_event_id'].drop_duplicates().sort_values().reset_index(drop=True)

# Padding events.
users_padded = pf.padding(users, ['conversation_id', 'session_id'])


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
    model = load_model('model_relative_representation_'+str(i)+'.h5')

    predictions = model.predict(users_padded)*post_filter
    
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
    model = load_model('model_relative_representation_'+str(i)+'.h5')

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
