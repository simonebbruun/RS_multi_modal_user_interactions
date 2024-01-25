import pandas as pd
import preprocessing_functions as pf
import numpy as np
from sklearn.model_selection import train_test_split
import gc
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import GRU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn import metrics


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
conversations = pd.concat((pd.read_csv('data_conversations_embedding_1.csv'),pd.read_csv('data_conversations_embedding_2.csv')),axis=0)
sessions = pd.read_csv('data_sessions.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting train events.
users = pf.train_test_split(users)
events = pf.event_type(users)
alpha = (len(events[events['event']=='conversation'])+len(events[events['event']=='intersection']))/len(events)
beta = (len(events[events['event']=='session'])+len(events[events['event']=='intersection']))/len(events)
users = users[users['purchase_event_id'].isin(events[events['event']=='intersection']['purchase_event_id'])]

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]


# Selecting conversations.
users_conv = users[~users['conversation_id'].isna()]
users_conv = users_conv[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')


# Predictions from conversation model.
predictions_conv = pf.pipeline_conversation(users_conv)

model = load_model('model_conversation_1.h5')
predictions_conv = model.predict(predictions_conv)


# Averaging embeddings.
users_conv = users_conv.drop(['event_number', 'conversation_id', 'sentence_number', 'sentence_speaker'], axis=1)
users_conv = users_conv.groupby('purchase_event_id').mean()
users_conv = users_conv.values


# Selecting sessions.
users_ses = users[~users['session_id'].isna()]
users_ses = users_ses[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')

# Predictions from session model.
predictions_ses = pf.pipeline_session(users_ses)

model = load_model('model_session_1.h5')
predictions_ses = model.predict(predictions_ses)


# Removing low frequent action_tags.
users_ses = pf.remove_low_frequent_features(users_ses, 'action_tags', ['purchase_event_id', 'session_id'], 'action_number', 'knowledge_distillation')

# Encoding sessions.
users_ses = pf.encoding(users_ses, features='action_tags',  id_columns=['session_id'], number_column='action_number', model_name='knowledge_distillation')

# Padding sessions.
users_ses = pf.padding(users_ses, ['session_id'])


# Binarizing purchase events and concatenating with predictions.
purchases = pf.binarize_items(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values
purchases = np.stack((purchases, predictions_conv, predictions_ses), axis = 1)

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


# Training and validation split.
train_x1, valid_x1, train_x2, valid_x2, train_y, valid_y, train_w, valid_w = train_test_split(users_conv, users_ses, purchases, post_filter, test_size=0.11, shuffle=False)

del users
del conversations
del sessions
del events
del users_conv
del users_ses
del purchases
del post_filter
del train_w
gc.collect()


''' Hyperparameter tuning.'''
# Defining loss function.
def knowledge_distillation_loss(y_true, y_pred):  
    y_pred = y_pred[:,0,:]
    y = y_true[:,0,:]
    z1 = y_true[:,1,:]
    z2 = y_true[:,2,:]
    
    loss = binary_crossentropy(y, y_pred) + alpha*binary_crossentropy(z1, y_pred) + beta*binary_crossentropy(z2, y_pred)
    return loss

set_random_seed(42)

for i in range(1,6):
    n_features1, n_features2, n_outputs = train_x1.shape[1], train_x2.shape[2], train_y.shape[2]
    epochs, batch_size, units, rate = 100, 256, 128, 0.4
    
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
    x = RepeatVector(3)(x)
    
    model = Model(inputs=[x1.input, x2.input], outputs=x)
    
    model.compile(loss=knowledge_distillation_loss, optimizer='adam')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('model_knowledge_distillation_'+str(i)+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    history = model.fit([train_x1, train_x2], train_y, validation_data=([valid_x1, valid_x2], valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])
    
    saved_model = load_model('model_knowledge_distillation_'+str(i)+'.h5', custom_objects={ 'knowledge_distillation_loss': knowledge_distillation_loss })
    valid_pred = saved_model.predict([valid_x1, valid_x2])[:,0,:]*valid_w
    
    print(metrics.roc_auc_score(valid_y[:,0,:], valid_pred))
