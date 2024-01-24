import pandas as pd
import numpy as np
import preprocessing_functions as pf
from sklearn.model_selection import train_test_split
import gc
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn import metrics


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
sessions = pd.read_csv('data_sessions.csv')
conversations = pd.read_csv('data_conversations_embedding.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting train events.
users = pf.train_test_split(users)

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]


# Selecting conversations.
users_conv = users[~users['conversation_id'].isna()]

users_conv = users_conv[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')
users_conv = users_conv.drop(['sentence_number', 'sentence_speaker'], axis=1)

# Averaging embeddings.
users_conv = users_conv.drop(['event_number', 'conversation_id'], axis=1)
users_conv = users_conv.groupby(['purchase_event_id'], as_index=False).mean()


# Selecting sessions.
users_ses = users[~users['session_id'].isna()]
users_ses = users_ses[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')

# Removing low frequent action_tags.
users_ses = pf.remove_low_frequent_features(users_ses, features='action_tags', id_columns=['purchase_event_id', 'session_id'], number_column='action_number', model_name='gen_conv', test = True)

# Encoding sessions.
users_ses = pf.encoding(users_ses, features='action_tags', id_columns=['session_id'], number_column='action_number', model_name='gen_conv', test = True)


''' Imputation. '''
# Imputing conversations.
users_gen_conv = users_ses[~users_ses['purchase_event_id'].isin(users_conv['purchase_event_id'])]
events = users_gen_conv['purchase_event_id'].drop_duplicates().sort_values().values
users_gen_conv = pf.padding(users_gen_conv, ['session_id'])
model = load_model('model_gen_conv.h5')
users_gen_conv = model.predict(users_gen_conv)

users_gen_conv = pd.DataFrame(users_gen_conv)
users_gen_conv.columns = users_conv.columns[1:]
users_gen_conv['purchase_event_id'] = events


# Imputing sessions.
users_gen_ses = users_conv[~users_conv['purchase_event_id'].isin(users_ses['purchase_event_id'])]
users_gen_ses = users_gen_ses.sort_values(by=['purchase_event_id']).reset_index(drop = True)
events = users_gen_ses['purchase_event_id'].values
users_gen_ses = users_gen_ses.drop(['purchase_event_id'], axis = 1).values
model = load_model('model_gen_ses.h5')
users_gen_ses = model.predict(users_gen_ses)
users_gen_ses = np.where(users_gen_ses<0.5,0,1)

users_gen_ses_1 = []
for i in range(len(users_gen_ses)):
    event = users_gen_ses[i]
    event = event[np.sum(event,axis=1)>0]
    event = pd.DataFrame(event)
    event.columns = users_ses.columns[3:]
    event['purchase_event_id'] = events[i]
    event['event_number'] = event.index+1
    users_gen_ses_1.append(event)
users_gen_ses_1 = pd.concat(users_gen_ses_1)


# Concatenating conversations with imputed conversations.
users_conv = pd.concat((users_conv, users_gen_conv))
users_conv = users_conv.sort_values(by=['purchase_event_id']).reset_index(drop = True)
users_conv = users_conv.drop(['purchase_event_id'], axis = 1).values

# Concatenating sessions with imputed sessions.
users_ses = users_ses.drop(['session_id'], axis=1)
users_ses = pd.concat((users_ses, users_gen_ses_1))
users_ses = pf.padding(users_ses, [])


# Binarizing purchase events.
purchases = pf.binarize_items(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


# Training and validation split.
train_x1, valid_x1, train_x2, valid_x2, train_y, valid_y, train_w, valid_w = train_test_split(users_conv, users_ses, purchases, post_filter, test_size=0.1, shuffle=False)

del users
del users_conv
del users_ses
del users_gen_conv
del users_gen_ses
del users_gen_ses_1
del events
del purchases
del post_filter
del train_w
gc.collect()


''' Model.'''
set_random_seed(42)

for i in range(1,6):
    n_features1, n_features2, n_outputs = train_x1.shape[1], train_x2.shape[2], train_y.shape[1]
    epochs, batch_size, units, rate = 100, 128, 256, 0.2
    
    input_x1 = Input(shape=(n_features1,))
    input_x2 = Input(shape=(None,n_features2))
    
    x1 = Dense(units)(input_x1)
    x1 = Model(inputs=input_x1, outputs=x1)
    
    x2 = Masking(mask_value=0.0)(input_x2)
    x2 = GRU(units)(x2)
    x2 = Model(inputs=input_x2, outputs=x2)
    
    combined = concatenate([x1.output, x2.output])
     
    x = Dense(units, activation='relu')(combined)
    x = Dropout(rate)(x)
    x = Dense(units, activation='relu')(x)
    x = Dense(n_outputs, activation='sigmoid')(x)
    
    model = Model(inputs=[x1.input, x2.input], outputs=x)
    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('model_gen_imputation_'+str(i)+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    history = model.fit([train_x1, train_x2], train_y, validation_data=([valid_x1, valid_x2], valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])
    
    saved_model = load_model('model_gen_imputation_'+str(i)+'.h5')
    valid_pred = saved_model.predict([valid_x1, valid_x2])*valid_w
    
    print(metrics.roc_auc_score(valid_y, valid_pred))



