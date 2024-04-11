#1.Data Preprocessing and Tokenization

import tensorflow as tf
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(texts, labels):
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens = True,
                            max_length = 64,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'tf'
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_masks), tf.convert_to_tensor(labels)

texts = ["This movie was great!", "I did not like the product."]
labels = [1, 0]  # 1 for positive, 0 for negative
input_ids, attention_masks, labels = preprocess_data(texts, labels)

#2.Model Training with TensorFlow

from transformers import TFBertForSequenceClassification
from transformers import BertConfig

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

model.compile(optimizer='adam', loss=model.compute_loss, metrics=['accuracy'])

model.fit([input_ids, attention_masks], labels, epochs=3, batch_size=32)

#3.Model Evaluation

results = model.evaluate([input_ids, attention_masks], labels)
print("Loss:", results[0])
print("Accuracy:", results[1])

"""These code snippets provide a complete implementation for sentiment analysis on reviews using TensorFlow and Hugging Face Transformers. 
The first code segment preprocesses the data and tokenizes it using BERT tokenizer. 
The second code segment trains the model using a pre-trained BERT model for sequence classification. 
Finally, the third code segment evaluates the model's performance on the provided data."""


