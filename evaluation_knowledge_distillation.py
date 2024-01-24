import pandas as pd
import preprocessing_functions as pf
import gc
import evaluation_functions as ef
from tensorflow.keras.models import load_model


''' Common preprocessing. '''
# Importing data.
users_union = pd.read_csv('data_users.csv')
conversations = pd.read_csv('data_conversations_embedding.csv')
sessions = pd.read_csv('data_sessions.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting test events.
users_union = pf.train_test_split(users_union, test = True)

conv_ses = pf.event_type(users_union)

users_union = users_union.merge(conv_ses, how = 'left')

purchases = purchases[purchases['purchase_event_id'].isin(users_union['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users_union['purchase_event_id'])]


# Binarizing purchase events.
purchases = pf.binarize_items(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


''' Preprocessing conversations. '''
# Selecting test events.
users_conv = users_union[users_union['event'] == 'conversation']
users_conv = users_conv[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')

events_conv = users_conv['purchase_event_id'].drop_duplicates().sort_values().reset_index(drop=True)


# Preprocessing conversations.
users_conv = pf.pipeline_conversation(users_conv)


''' Preprocessing sessions. '''
# Selecting test events.
users_ses = users_union[users_union['event'] == 'session']
users_ses = users_ses[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')

events_ses = users_ses['purchase_event_id'].drop_duplicates().sort_values().reset_index(drop=True)


# Preprocessing sessions.
users_ses = pf.pipeline_session(users_ses)


''' Preprocessing intersections. '''
users_int = users_union[users_union['event'] == 'intersection']
events_int = users_int['purchase_event_id'].drop_duplicates().sort_values().reset_index(drop=True)

# Conversations.
users_int_conv = users_int[~users_int['conversation_id'].isna()]
users_int_conv = users_int_conv[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')

# Averaging embeddings.
users_int_conv = users_int_conv.drop(['event_number', 'conversation_id', 'sentence_number', 'sentence_speaker'], axis=1)
users_int_conv = users_int_conv.groupby('purchase_event_id').mean()
users_int_conv = users_int_conv.values

# Sessions.
users_int_ses = users_int[~users_int['session_id'].isna()]
users_int_ses = users_int_ses[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')

# Removing low frequent action_tags.
users_int_ses = pf.remove_low_frequent_features(users_int_ses, 'action_tags', ['purchase_event_id', 'session_id'], 'action_number', 'knowledge_distillation', test = True)

# Encoding sessions.
users_int_ses = pf.encoding(users_int_ses, features='action_tags',  id_columns=['session_id'], number_column='action_number', model_name='knowledge_distillation', test = True)

# Padding sessions.
users_int_ses = pf.padding(users_int_ses, ['session_id'])


del conversations
del sessions
del users_union
del users_int
gc.collect()


''' Evaluation. '''
def knowledge_distillation_loss(y_true, y_pred):  
    loss = None
    return loss

# Average of 5 runs.
k = 3
    
evaluation_list = []
evaluation_list_union = []
for i in range(1,6):
    model_conv = load_model('model_conversation_'+str(i)+'.h5')
    model_ses = load_model('model_session_'+str(i)+'.h5')
    model_int = load_model('model_knowledge_distillation_'+str(i)+'.h5', custom_objects={ 'knowledge_distillation_loss': knowledge_distillation_loss})
    
    predictions = ef.predict_knowledge_distillation(dataframes_user = [users_conv, users_ses, [users_int_conv, users_int_ses]], dataframes_event = [events_conv, events_ses, events_int], models=[model_conv, model_ses, model_int])
    events = predictions['purchase_event_id']
    predictions = predictions.drop(['purchase_event_id'], axis=1).values*post_filter    
    
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
    model_conv = load_model('model_conversation_'+str(i)+'.h5')
    model_ses = load_model('model_session_'+str(i)+'.h5')
    model_int = load_model('model_knowledge_distillation_'+str(i)+'.h5', custom_objects={ 'knowledge_distillation_loss': knowledge_distillation_loss})
    
    predictions = ef.predict_knowledge_distillation(dataframes_user = [users_conv, users_ses, [users_int_conv, users_int_ses]], dataframes_event = [events_conv, events_ses, events_int], models=[model_conv, model_ses, model_int])
    events = predictions['purchase_event_id']
    predictions = predictions.drop(['purchase_event_id'], axis=1).values*post_filter    
    
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