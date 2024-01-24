import pandas as pd
import preprocessing_functions as pf
import numpy as np
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
sessions = pd.read_csv('data_sessions.csv')
conversations = pd.read_csv('data_conversations_keyword.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting train events.
users = pf.train_test_split(users)

users_conversation = users[~users['conversation_id'].isna()].merge(conversations, how = 'left')
users_conversation = users_conversation.drop(['sentence_speaker'], axis=1)
users_conversation = users_conversation.rename(columns = {'sentence_number' : 'number'})
users_session = users[~users['session_id'].isna()].merge(sessions, how = 'left')
users_session = users_session.rename(columns = {'action_number' : 'number', 'action_tags' : 'keywords'})
users = pd.concat([users_conversation, users_session], axis=0)
users = users[~users['keywords'].isna()]
users = users[users['keywords']!='']

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]


# Removing low frequent keywords.
users = pf.remove_low_frequent_features(users, features='keywords', id_columns=['purchase_event_id', 'conversation_id', 'session_id'], number_column='number', model_name='keyword')

# Encoding conversations and sessions.
users['conversation_ind'] = np.where(users['session_id'].isna(), 1, 0)
users = pf.encoding(users, features='keywords', id_columns=['conversation_id', 'session_id'], number_column='number', model_name='keyword')

# Padding events.
users = pf.padding(users, ['conversation_id', 'session_id'])


# Binarizing purchase events.
purchases = pf.binarize_items(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


# Training and validation split.
train_x, valid_x, train_y, valid_y, train_w, valid_w = train_test_split(users, purchases, post_filter, test_size=0.1, shuffle=False)

del users
del users_conversation
del users_session
del conversations
del sessions
del purchases
del post_filter
del train_w
gc.collect()


''' Model.'''
set_random_seed(42)

for i in range(1,6):
    n_features, n_outputs = train_x.shape[2], train_y.shape[1]
    epochs, batch_size, units, rate = 100, 512, 256, 0.2
    
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=(None,n_features)))
    model.add(GRU(units, return_sequences=False))
    model.add(Dropout(rate))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('model_keyword_'+str(i)+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])
    
    saved_model = load_model('model_keyword_'+str(i)+'.h5')
    valid_pred = saved_model.predict(valid_x)*valid_w
    
    print(metrics.roc_auc_score(valid_y, valid_pred))

