import pandas as pd
import preprocessing_functions as pf
from sklearn.model_selection import train_test_split
import gc
from tensorflow.keras.utils import set_random_seed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
conversations = pd.concat((pd.read_csv('data_conversations_embedding_1.csv'),pd.read_csv('data_conversations_embedding_2.csv')),axis=0)
purchases = pd.read_csv('data_purchase_events.csv')
post_filter = pd.read_csv('data_post_filter.csv')


# Selecting train events.
users = pf.train_test_split(users)
users = users[~users['conversation_id'].isna()]

purchases = purchases[purchases['purchase_event_id'].isin(users['purchase_event_id'])]
post_filter = post_filter[post_filter['purchase_event_id'].isin(users['purchase_event_id'])]


# Averaging embeddings.
users = users[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')
users = users.drop(['sentence_number', 'sentence_speaker'], axis=1)


users = users.groupby(['purchase_event_id', 'event_number', 'conversation_id'], as_index=False).mean()
users = users.sort_values(by=['purchase_event_id']).reset_index(drop=True)


# Binarizing purchase events.
purchases = pf.binarize_items(purchases)
purchases = users[['purchase_event_id']].merge(purchases)
purchases = purchases.drop(['purchase_event_id'], axis=1).values


# Binarizing post filter.
post_filter = pf.binarize_items(post_filter)
post_filter = users[['purchase_event_id']].merge(post_filter)
post_filter = post_filter.drop(['purchase_event_id'], axis=1).values


users = users.drop(['purchase_event_id', 'event_number', 'conversation_id'], axis=1).values


# Training and validation split.
train_x, valid_x, train_y, valid_y, train_w, valid_w = train_test_split(users, purchases, post_filter, test_size=0.1, shuffle=False)

del users
del conversations
del purchases
del post_filter
del train_w
gc.collect()


''' Model.'''
set_random_seed(42)

n_features, n_outputs = train_x.shape[1], train_y.shape[1]
epochs, batch_size, units, rate = 100, 128, 256, 0.4

model_input = Input(shape=(n_features))
encoder = Dense(units, activation='tanh')(model_input)
decoder = Dropout(rate)(encoder)
decoder = Dense(units, activation='relu')(decoder)
decoder = Dense(n_outputs, activation='sigmoid')(decoder)

model = Model(inputs=model_input, outputs=decoder)

model.compile(loss='binary_crossentropy', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model_rel_conversation.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True)

history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])

model.load_weights('model_rel_conversation.h5')
valid_pred = model.predict(valid_x)*valid_w

print(metrics.roc_auc_score(valid_y, valid_pred))

encoder_model = Model(inputs=model_input, outputs=encoder)
encoder_model.save('encoder_conversation.h5')
