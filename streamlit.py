#toeknized text
import transformers
from transformers import AutoTokenizer
from transformers import  DistilBertForTokenClassification

import torch
import torch.nn as nn

import torch
import re

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

class DistilbertNER(nn.Module):
  """
  Implement NN class based on distilbert pretrained from Hugging face.
  Inputs : 
    tokens_dim : int specifyng the dimension of the classifier
  """
  
  def __init__(self, tokens_dim):
    super(DistilbertNER,self).__init__()
    
    if type(tokens_dim) != int:
            raise TypeError('Please tokens_dim should be an integer')

    if tokens_dim <= 0:
          raise ValueError('Classification layer dimension should be at least 1')

    self.pretrained = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels = tokens_dim) #set the output of each token classifier = unique_lables


  def forward(self, input_ids, attention_mask, labels = None): #labels are needed in order to compute the loss
    """
  Forwad computation of the network
  Input:
    - inputs_ids : from model tokenizer
    - attention :  mask from model tokenizer
    - labels : if given the model is able to return the loss value
  """

    #inference time no labels
    if labels == None:
      out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask )
      return out

    out = self.pretrained(input_ids = input_ids, attention_mask = attention_mask , labels = labels)
    return out



# Define the file path where the model is saved
model_save_path = "distilbert_ner_model_meta.pth"

# Load the model data
model_data = torch.load(model_save_path, map_location=torch.device('cpu'))

# Extract the model's state dictionary and metadata
model_state_dict = model_data["model_state_dict"]
metadata = model_data["metadata"]
idx2tag = metadata["idx2tag"]
tag2idx = metadata["tag2idx"]

# Load the model class
model = DistilbertNER(len(metadata["unique_tags"]))  # Assuming DistilbertNER is the class used to define your model

# Load the model's state dictionary
model.load_state_dict(model_state_dict)



def align_word_ids(texts):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

def evaluate_one_text(model, sentence):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    sentence_processed, pincode = preprocess_user_input(sentence)
    sentence =  sentence_processed

    text = tokenizer(sentence, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [idx2tag[i] for i in predictions]

    if pincode:
        prediction_label.append('pincode')
        sentence_processed += ' ' + pincode[0]
        

    return sentence_processed, prediction_label

def preprocess_user_input(sentence):
    # Lowercase the input
    sentence = sentence.lower()
    # Remove commas
    sentence = sentence.replace(',', ' ')
    # Extract pincode using regex
    pincode = re.findall(r'\b\d{6}\b', sentence)
    # Remove pincode from the sentence
    sentence = re.sub(r'\b\d{6}\b', '', sentence)
    # Join the remaining text
    sentence = ' '.join(sentence.split())
    return sentence, pincode


# create a streamlit app and return a json onject with tags and its words in sentence
# Path: final_model/streamlit.py
import streamlit as st
import re
import torch
def main():
    st.title("Named Entity Recognition with DistilBERT")

    # Get user input
    sentence = st.text_input("Enter a sentence:")

    if st.button("Predict"):
        # Perform prediction
        sentence,predictions = evaluate_one_text(model, sentence)
        
        # Format predictions into JSON
        predictions_json = {"sentence": sentence, "predictions": predictions}

        # Display predictions
        st.json(predictions_json)

if __name__ == "__main__":
    main()
