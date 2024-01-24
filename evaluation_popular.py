import pandas as pd
import preprocessing_functions as pf
import evaluation_functions as ef
from pickle import load
import numpy as np


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting test events.
users = pf.train_test_split(users, test = True)

conv_ses = pf.event_type(users)

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]

events = users['purchase_event_id'].drop_duplicates().sort_values().reset_index(drop=True)


# Binarizing purchase events.
purchases = pf.binarize_items(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


''' Evaluation. '''
# Average of 5 runs.
k = 3
    
model = load(open('model_popular.npy', 'rb'))

predictions = np.tile(model, (len(events),1))*post_filter

hit = ef.hit(predictions, purchases, k)
average_precision = ef.average_precision(predictions, purchases, k)

evaluation_avg = pd.DataFrame({'purchase_event_id' : events, 'hit_rate' : hit, 'mean_average_precision' : average_precision})
evaluation_avg = conv_ses.merge(evaluation_avg, how = 'left')

evaluation_avg_union = pd.DataFrame(evaluation_avg[['hit_rate', 'mean_average_precision']].mean()).transpose()
evaluation_avg_union['event'] = 'union'
evaluation_avg = evaluation_avg[['event', 'hit_rate', 'mean_average_precision']].groupby('event', as_index = False).mean()
evaluation_avg = pd.concat((evaluation_avg, evaluation_avg_union))


# Varying thresholds.
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