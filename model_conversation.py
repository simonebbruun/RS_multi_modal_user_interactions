import pandas as pd
import preprocessing_functions as pf
from sklearn.model_selection import train_test_split
import gc
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn import metrics


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
conversations = pd.read_csv('data_conversations_embedding.csv')
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting train events.
users = pf.train_test_split(users)
users = users[~users['conversation_id'].isna()]

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]


# Averaging embeddings.
users = users[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')
users = users.drop(['event_number', 'conversation_id', 'sentence_number', 'sentence_speaker'], axis=1)

users = users.groupby('purchase_event_id').mean()
users = users.values


# Binarizing purchase events.
purchases = pf.binarize_items(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values

# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


# Training and validation split.
train_x, valid_x, train_y, valid_y, train_w, valid_w = train_test_split(users, purchases, post_filter, test_size=0.1, shuffle=False)

del users
del purchases
del post_filter
del train_w
gc.collect()


''' Model.'''
set_random_seed(42)

for i in range(1,6):
    n_features, n_outputs = train_x.shape[1], train_y.shape[1]
    epochs, batch_size, units, rate = 100, 512, 64, 0.2
    
    model = Sequential()
    model.add(Dense(units, input_dim=n_features))
    model.add(Dropout(rate))
    model.add(Dense(units, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    mc = ModelCheckpoint('model_conversation_'+str(i)+'.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    
    history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])
    
    saved_model = load_model('model_conversation_'+str(i)+'.h5')
    valid_pred = saved_model.predict(valid_x)*valid_w
    
    print(metrics.roc_auc_score(valid_y, valid_pred))

