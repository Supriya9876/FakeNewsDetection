import numpy as np
import pycaret
import transformers
from transformers import AutoModel, BertTokenizerFast
import torch
import torch.nn as nn
# specify GPU
device = torch.device("cuda")


### Define Model Architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
    def forward(self, sent_id, mask):           # define the forward pass
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x


### Load pretrained BERT Model
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BERT_Arch(bert)

### Model Predictions

# load the model
path = 'models/Bert_model.pt'
model.load_state_dict(torch.load(path))

def predict(user_text):
  # testing on unseen data
  unseen_news_text = []
  unseen_news_text.append(user_text)

  # tokenize and encode sequences in the test set
  MAX_LENGHT = 50
  tokens_unseen = tokenizer.batch_encode_plus(
      unseen_news_text,
      max_length = MAX_LENGHT,
      pad_to_max_length=True,
      truncation=True
  )

  unseen_seq = torch.tensor(tokens_unseen['input_ids'])
  unseen_mask = torch.tensor(tokens_unseen['attention_mask'])

  with torch.no_grad():
    preds = model(unseen_seq, unseen_mask)
    preds = preds.detach().cpu().numpy()

  preds = np.argmax(preds, axis = 1)

  return preds[0]



  """
  
  Note : This is our prediction model which is based on BERT Model.
  
  """