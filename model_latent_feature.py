import pandas as pd
import preprocessing_functions as pf
import numpy as np
from sklearn.model_selection import train_test_split
import gc
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn import metrics


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
sessions = pd.read_csv('data_sessions.csv')
conversations = pd.concat((pd.read_csv('data_conversations_embedding_1.csv'),pd.read_csv('data_conversations_embedding_2.csv')),axis=0)
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
users_conv = users_conv.groupby(['purchase_event_id', 'event_number', 'conversation_id'], as_index=False).mean()
users_conv['conversation_ind'] = 1


# Selecting sessions.
users_ses = users[~users['session_id'].isna()]
users_ses = users_ses[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')


# Removing low frequent action_tags.
users_ses = pf.remove_low_frequent_features(users_ses, features='action_tags', id_columns=['purchase_event_id', 'session_id'], number_column='action_number', model_name='latent_feature')

# Encoding sessions.
users_ses = pf.encoding(users_ses, features='action_tags', id_columns=['session_id'], number_column='action_number', model_name='latent_feature')
users_ses['conversation_ind'] = 0


# Setting sample weight.
weights = pf.event_type(users)
distribution = weights.groupby(['event']).size()/len(weights)
w_conversation = 2
w_intersection = (w_conversation*distribution['conversation']+distribution['session'])/(1-distribution['intersection'])
weights['weight'] = np.where(weights['event']=='conversation',w_conversation,
                             np.where(weights['event']=='session',1,
                                      w_intersection))
weights = weights['weight'].values


# Joining conversations and sessions.
users = pd.concat([users_conv, users_ses], axis = 0)
users = users.fillna(0)


# Padding conversations and sessions.
users = pf.padding(users, ['conversation_id', 'session_id'])


# Binarizing purchase events.
purchases = pf.binarize_items(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


# Training and validation split.
train_x1, valid_x1, train_x2, valid_x2, train_x3, valid_x3, train_y, valid_y, train_w1, valid_w1, train_w2, valid_w2 = train_test_split(users[:,:,0:768], users[:,:,769:824], users[:,:,768:769], purchases, weights, post_filter, test_size=0.1, shuffle=False)

del users
del conversations
del sessions
del users_conv
del users_ses
del purchases
del post_filter
del weights
del train_w2
gc.collect()


''' Model.'''
set_random_seed(42)

for i in range(1,6):
    n_features1, n_features2, n_features3, n_outputs = train_x1.shape[2], train_x2.shape[2], train_x3.shape[2], train_y.shape[1]
    epochs, batch_size, units, rate = 100, 512, 256, 0.3
    
    input_x1 = Input(shape=(None,n_features1))
    input_x2 = Input(shape=(None,n_features2))
    input_x3 = Input(shape=(None,n_features3))
    
    x1 = TimeDistributed(Masking(mask_value=0.0))(input_x1)
    x1 = TimeDistributed(Dense(round(units/2), activation='tanh'))(x1)
    x1 = Model(inputs=input_x1, outputs=x1)
    
    x2 = TimeDistributed(Masking(mask_value=0.0))(input_x2)
    x2 = TimeDistributed(Dense(round(units/2), activation='tanh'))(x2)
    x2 = Model(inputs=input_x2, outputs=x2)
    
    combined1 = add([x1.output, x2.output])
    combined2 = concatenate([combined1, input_x3])
     
    x = Masking(mask_value=0.0)(combined2) 
    x = GRU(units)(x)
    x = Dropout(rate)(x)
    x = Dense(units, activation='relu')(x)
    x = Dense(n_outputs, activation='sigmoid')(x)
    
    model = Model(inputs=[x1.input, x2.input, input_x3], outputs=x)
    
    model.compile(loss='binary_crossentropy', optimizer='adam', weighted_metrics=[])
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('model_latent_feature_'+str(i)+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    history = model.fit([train_x1, train_x2, train_x3], train_y, validation_data=([valid_x1, valid_x2, valid_x3], valid_y, valid_w1), sample_weight=train_w1, epochs=epochs, batch_size=batch_size, callbacks=[es, mc])
    
    saved_model = load_model('model_latent_feature_'+str(i)+'.h5')
    valid_pred = saved_model.predict([valid_x1, valid_x2, valid_x3])*valid_w2
    
    print(metrics.roc_auc_score(valid_y, valid_pred))
