import pandas as pd
import preprocessing_functions as pf
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


''' Data preprocessing. '''
# Importing data.
users = pd.read_csv('data_users.csv')
conversations = []
for i in range(1,108):
    conversations.append(pd.read_csv('data_conversations_embedding/data_conversations_embedding_'+str(i)+'.csv'))
conversations = pd.concat(conversations)
sessions = pd.read_csv('data_sessions.csv')


# Selecting train events.
users = pf.train_test_split(users)
events = pf.event_type(users)
users = users[users['purchase_event_id'].isin(events[events['event']=='intersection']['purchase_event_id'])]


# Selecting conversations.
users_conv = users[~users['conversation_id'].isna()]
users_conv = users_conv[['purchase_event_id', 'event_number', 'conversation_id']].merge(conversations, how = 'left')


# Averaging embeddings.
users_conv = users_conv.drop(['event_number', 'conversation_id', 'sentence_number', 'sentence_speaker'], axis=1)
users_conv = users_conv.groupby('purchase_event_id').mean()
users_conv = users_conv.values


# Selecting sessions.
users_ses = users[~users['session_id'].isna()]
users_ses = users_ses[['purchase_event_id', 'event_number', 'session_id']].merge(sessions, how = 'left')


# Removing low frequent action_tags.
users_ses = pf.remove_low_frequent_features(users_ses, 'action_tags', ['purchase_event_id', 'session_id'], 'action_number', 'gen_conv')

# Encoding sessions.
users_ses = pf.encoding(users_ses, features='action_tags',  id_columns=['session_id'], number_column='action_number', model_name='gen_conv')

# Padding sessions.
users_ses = pf.padding(users_ses, ['session_id'])


# Training and validation split.
train_x, valid_x, train_y, valid_y = train_test_split(users_ses, users_conv, test_size=0.1, shuffle=False)

del users
del conversations
del sessions
del events
del users_conv
del users_ses
gc.collect()


''' Model.'''
set_random_seed(42)

n_features, n_outputs = train_x.shape[2], train_y.shape[1]
epochs, batch_size, units, rate = 100, 128, 64, 0.4

model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(None,n_features)))
model.add(GRU(units, return_sequences=False))
model.add(Dropout(rate))
model.add(Dense(units, activation='relu'))
model.add(Dense(n_outputs, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('model_gen_conv.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y), epochs=epochs, batch_size=batch_size, callbacks=[es, mc])

saved_model = load_model('model_gen_conv.h5')
valid_mse = saved_model.evaluate(valid_x, valid_y)

print(valid_mse)
