import numpy as np
import pandas as pd


def predict_late_fusion(dataframes_user, dataframes_event, dataframe_conv_ses, models):
    # Predictions conversation.
    predictions_conv = models[0].predict(dataframes_user[0])
    predictions_conv = pd.concat([dataframes_event[0], pd.DataFrame(predictions_conv)], axis = 1)
    predictions_conv = predictions_conv.merge(dataframe_conv_ses[['purchase_event_id', 'event']], how = 'left')
    
    # Predictions session. 
    predictions_ses = models[1].predict(dataframes_user[1])
    predictions_ses = pd.concat([dataframes_event[1], pd.DataFrame(predictions_ses)], axis = 1)
    predictions_ses = predictions_ses.merge(dataframe_conv_ses[['purchase_event_id', 'event']], how = 'left')
    
    # Predictions intersection (fusion).
    intersections_conv = predictions_conv[predictions_conv['event'] == 'intersection'].drop(['purchase_event_id', 'event'], axis=1)
    intersections_conv = intersections_conv.values
    intersections_ses = predictions_ses[predictions_ses['event'] == 'intersection'].drop(['purchase_event_id', 'event'], axis=1)
    intersections_ses = intersections_ses.values
    
    events = predictions_conv[predictions_conv['event'] == 'intersection']['purchase_event_id'].sort_values().reset_index(drop=True)
    
    predictions_intersection = 0.5*intersections_conv+0.5*intersections_ses
    predictions_intersection = pd.concat([events, pd.DataFrame(predictions_intersection)], axis=1)
    
    # Predictions union.
    predictions = pd.concat([predictions_conv[predictions_conv['event'] == 'conversation'].drop(['event'], axis=1), predictions_ses[predictions_ses['event'] == 'session'].drop(['event'], axis=1), predictions_intersection], axis=0)
    predictions = predictions.sort_values(by='purchase_event_id').reset_index(drop=True) 
    
    return predictions


def predict_knowledge_distillation(dataframes_user, dataframes_event, models):
    # Predictions conversation.
    predictions_conv = models[0].predict(dataframes_user[0])
    predictions_conv = pd.concat([dataframes_event[0], pd.DataFrame(predictions_conv)], axis = 1)

    # Predictions session. 
    predictions_ses = models[1].predict(dataframes_user[1])
    predictions_ses = pd.concat([dataframes_event[1], pd.DataFrame(predictions_ses)], axis = 1)

    # Predictions intersection (knowledge distillation).
    predictions_int = models[2].predict(dataframes_user[2])[:,0,:]
    predictions_int = pd.concat([dataframes_event[2], pd.DataFrame(predictions_int)], axis = 1)

    # Predictions union.
    predictions = pd.concat([predictions_conv, predictions_ses, predictions_int], axis=0)
    predictions = predictions.sort_values(by='purchase_event_id').reset_index(drop=True)        
    
    return predictions
    

def hit(predictions, test_set, k):
    n_obs = test_set.shape[0]
    rank = (-predictions).argsort()
    top_k_recommendations = rank[:,0:k]
    hit = np.empty([n_obs*k]).reshape(n_obs, k)
    for i in range(n_obs):
        hit[i,:] = test_set[i,top_k_recommendations[i,:]]
    hit = np.max(hit, axis=1)
    return hit


def precision(predictions, test_set, k):
    n_obs = test_set.shape[0]
    rank = (-predictions).argsort()
    top_k_recommendations = rank[:,0:k]
    labels = [np.nonzero(t)[0] for t in test_set]
    true_labels_captured = np.empty([n_obs])
    for i in range(n_obs):
        true_labels_captured[i] = len(np.intersect1d(top_k_recommendations[i,:],labels[i]))

    true_labels = np.empty([n_obs])
    for i in range(n_obs):
        true_labels[i] = len(labels[i])

    precision = true_labels_captured/k
    return precision 


def recall(predictions, test_set, k):
    n_obs = test_set.shape[0]
    rank = (-predictions).argsort()
    top_k_recommendations = rank[:,0:k]
    labels = [np.nonzero(t)[0] for t in test_set]
    true_labels_captured = np.empty([n_obs])
    for i in range(n_obs):
        true_labels_captured[i] = len(np.intersect1d(top_k_recommendations[i,:],labels[i]))

    true_labels = np.empty([n_obs])
    for i in range(n_obs):
        true_labels[i] = len(labels[i])

    recall = true_labels_captured/true_labels
    return recall


def reciprocal_rank(predictions, test_set, k):
    rank = (-predictions).argsort()
    ranked_items = rank.argsort()
    relevant_items = np.where(test_set == 1, ranked_items, np.nan)
    relevant_items1 = np.where(relevant_items >= k, np.nan, relevant_items)
    min_rank = np.nanmin(relevant_items1, axis=1)
    rr = 1/(min_rank+1)
    rr = np.nan_to_num(rr)
    return rr


def average_precision(predictions, test_set, k):
    n_obs = test_set.shape[0]
    n_items = test_set.shape[1]
    labels = [np.nonzero(t)[0] for t in test_set]
    precision_at_j = np.empty([n_obs,n_items])
    for j in range(n_items):
        top_j_recommendations = (-predictions).argsort()[:,0:(j+1)]
        
        true_labels_captured = np.empty([n_obs,1])
        for i in range(n_obs):
            true_labels_captured[i] = len(np.intersect1d(top_j_recommendations[i,:],labels[i]))
            
        precision_at_j[:,j:(j+1)] = true_labels_captured/(j+1)
        
    relevant_j = np.empty([n_obs,n_items])
    for j in range(n_items):
        top_j_recommendations = (-predictions).argsort()[:,0:(j+1)]
        for i in range(n_obs):
            relevant_j[i,j] = np.where(len(np.intersect1d(top_j_recommendations[i,j],labels[i]))==1,1,0)

    L = np.empty([n_obs], dtype='int')
    for i in range(n_obs):
        L[i] = min(k,len(labels[i]))

    ap = np.empty([n_obs])
    for i in range(n_obs):
        ap[i] = sum(np.multiply(precision_at_j[:,0:k],relevant_j[:,0:k])[i,:])/L[i]
    return ap