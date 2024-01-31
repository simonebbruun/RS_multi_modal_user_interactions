# RS_multi_modal_user_interactions
This repository contains the data and source code for **Utilizing Multi-Modal User Interactions for Personalized Item Recommendations**.

## Requirements

- Python
- NumPy
- Pandas
- TensorFlow
- Scikit-learn
- Pickle
- Matplotlib


## Dataset

We publish a real-world dataset from the insurance domain with multi-modal user interactions that can be used in recommendation models. The dataset is anonymized.  
Download the files: data_users.csv, data_conversations_keyword.csv, data_sessions.csv, data_purchase_events.csv, data_post_filter.csv   
and the folder: data_conversations_embedding


## Dataset Format
There are 6 different datasets.

### data_users.csv

This data contains the users. Each user has had one or more purchase events with conversations and/or web sessions prior to that purchase. The data contains 5 columns:
- user_id. The ID of a user.
- purchase_event_id. The ID of a purchase event.
- conversation_id. The ID of a conversation.
- session_id. The ID of a web session.
- event_number. A number specifying the order of conversations/web sessions.

### data_conversations_keyword.csv

This data contains the conversations that the user had prior to the user's purchase event. Each conversation consists of multiple sentences represented with keywords. The data contains 4 columns:
- conversation_id. The ID of a conversation.
- sentence_number. A number specifying the order of sentences.
- sentence_speaker. The speaker of the sentence (user or agent).
- keywords. List with the IDs of the keywords in the sentence.

### data_conversations_embedding_(1-107).csv

This data is split into multiple files due to file size limitations.
The data contains the conversations that the user had prior to the user's purchase event. Each conversation consists of multiple sentences represented with text embeddings. The data contains 771 columns:
- conversation_id. The ID of a conversation.
- sentence_number. A number specifying the order of sentences.
- sentence_speaker. The speaker of the sentence (user or agent).
- embedding_1 - embedding_768. Text embeddings computed with a pre-trained language-specific BERT model.

### data_sessions.csv

This data contains the web sessions that the user made prior to the user's purchase event. Each web session consists of multiple actions. The data contains 3 columns:
- session_id. The ID of a web session.
- action_number. A number specifying the order of actions.
- action_tags. List with the IDs of the section, object and type of an action.

### data_purchase_events.csv

This data contains the purchase events. Each event consists of one or more item purchases made by the same user. The data contains 2 columns:
- purchase_event_id. The ID of a purchase event.   
- item_id. The ID of an item.   

### data_post_filter.csv

This data contains the items that were possible for the user to buy at the time of the user's purchase event. The data contains 2 columns:
- purchase_event_id. The ID of a purchase event.   
- item_id. The ID of an item.   


## Usage

1. Train and validate the models using  
   model_popular.py  
   model_conversation.py  
   model_session.py  
   model_late_fusion.py  
   model_knowledge_distillation.py  
   model_generative_imputation_step_1.py  
   model_generative_imputation_step_2.py  
   model_generative_imputation_step_3.py  
   model_neutral_imputation.py  
   model_keyword.py  
   model_latent_feature.py  
   model_relative_representation_step_1.py  
   model_relative_representation_step_2.py  
   model_relative_representation_step_3.py  
3. Evaluate the models over the test set using  
   evaluation_popular.py  
   evaluation_conversation.py  
   evaluation_session.py  
   evaluation_late_fusion.py  
   evaluation_knowledge_distillation.py  
   evaluation_generative_imputation.py  
   evaluation_neutral_imputation.py  
   evaluation_keyword.py  
   evaluation_latent_feature.py  
   evaluation_relative_representation.py  
