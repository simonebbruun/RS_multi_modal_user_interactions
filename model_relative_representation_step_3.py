import pandas as pd
import preprocessing_functions as pf
import numpy as np
from pickle import dump
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import gc
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn import metrics


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


# Selecting train events.
users = pf.train_test_split(users)

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
purchases = pf.binarize_items(purchases)
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]

# Selecting anchor_events
n_anchors = 125
anchor_events = pf.event_type(users.groupby(['purchase_event_id']).filter(lambda x: len(x) == 2))
anchor_events = anchor_events[anchor_events['event']=='intersection']
anchor_events = anchor_events.merge(purchases)
weights = purchases.drop(['purchase_event_id'],axis=1).sum(axis=0)/len(purchases)
weights = weights/weights.sum()
weights = np.maximum(round(weights*n_anchors),1).astype(int)
for i in weights.index:
    anchor_events[i] = anchor_events[i]*weights[i]
anchor_events['class'] = anchor_events[weights.index].replace(0,99).idxmin(axis='columns')
anchor_samples = []
for i in weights.index:
    anchor_samples.append(anchor_events[anchor_events['class']==i].sample(weights[i], random_state = 42))
anchor_samples = np.vstack(anchor_samples)[:,0]

# Selecting conversations.
users_conv = users[~users['conversation_id'].isna()]

users_conv = users_conv[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')
users_conv = users_conv.drop(['sentence_number', 'sentence_speaker'], axis=1)

# Averaging embeddings.
users_conv = users_conv.groupby(['purchase_event_id', 'event_number', 'conversation_id'], as_index=False).mean()
users_conv['conversation_ind'] = 1


# Computing relative representations of conversations.
encoder = load_model('encoder_conversation.h5')
encodings = encoder.predict(users_conv.drop(['purchase_event_id','event_number','conversation_id', 'conversation_ind'], axis=1))

anchor_indices = users_conv[users_conv['purchase_event_id'].isin(anchor_samples)].index
anchor_save = encodings[anchor_indices,:]
dump(anchor_save, open('anchors_conversation.npy', 'wb'))

relative_representations = []
for a in anchor_indices:
    relative_representations.append(cosine_similarity(encodings,encodings[None,a]))
relative_representations = np.hstack(relative_representations)

users_conv = pd.concat([users_conv[['purchase_event_id','event_number','conversation_id', 'conversation_ind']], pd.DataFrame(relative_representations, columns = ['rel_' + str(i + 1) for i in range(relative_representations.shape[1])])], axis=1)


# Selecting sessions.
users_ses = users[~users['session_id'].isna()]
users_ses = users_ses[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')

# Removing low frequent action_tags.
users_ses = pf.remove_low_frequent_features(users_ses, features='action_tags', id_columns=['purchase_event_id', 'session_id'], number_column='action_number', model_name='relative_representation')

# Encoding sessions.
users_ses = pf.encoding(users_ses, features='action_tags', id_columns=['session_id'], number_column='action_number', model_name='relative_representation')
users_ses['conversation_ind'] = 0


# Computing relative representations of sessions.
encoder = load_model('encoder_session.h5')
encodings = encoder.predict(users_ses.drop(['purchase_event_id','event_number','session_id', 'conversation_ind'], axis=1))

anchor_indices = users_ses[users_ses['purchase_event_id'].isin(anchor_samples)].index
anchor_save = encodings[anchor_indices,:]
dump(anchor_save, open('anchors_session.npy', 'wb'))

relative_representations = []
for a in anchor_indices:
    relative_representations.append(cosine_similarity(encodings,encodings[None,a]))
relative_representations = np.hstack(relative_representations)

users_ses = pd.concat([users_ses[['purchase_event_id','event_number','session_id', 'conversation_ind']], pd.DataFrame(relative_representations, columns = ['rel_' + str(i + 1) for i in range(relative_representations.shape[1])])], axis=1)


# Joining conversations and sessions.
users = pd.concat([users_conv, users_ses], axis = 0)


# Padding conversations and sessions.
users = pf.padding(users, ['conversation_id', 'session_id'])


# Binarizing purchase events.
purchases = purchases.drop(['purchase_event_id'], axis=1).values

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


# Training and validation split.
train_x, valid_x, train_y, valid_y, train_w, valid_w = train_test_split(users, purchases, post_filter, test_size=0.1, shuffle=False)

del users
del conversations
del sessions
del users_conv
del users_ses
del purchases
del post_filter
del train_w
del encodings
del relative_representations
gc.collect()


''' Model.'''
n_features, n_outputs = train_x.shape[2], train_y.shape[1]
epochs, batch_size, units, rate = 100, 256, 256, 0.3

set_random_seed(42)
for i in range(1,6):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(None,n_features)))
    model.add(GRU(units, return_sequences=False))
    model.add(Dropout(rate))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('model_relative_representation_'+str(i)+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])
    
    saved_model = load_model('model_relative_representation_'+str(i)+'.h5')
    valid_pred = saved_model.predict(valid_x)*valid_w
    
    print(metrics.roc_auc_score(valid_y, valid_pred))
