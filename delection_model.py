from transformers import AutoModel
import torch.nn as nn
import torch

class MultiLingualClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(MultiLingualClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits



def predict_language(text, model, tokenizer, device, max_length=128):
    """
        detect the language of the input text 
        the language supported:
            English, French, Spanish, Portugeese, 
            Italian, Russian, Sweedish, Malayalam, Dutch,
            Arabic, Turkish, German, Tamil, Danish, Kannada, Greek, Hindi
        
        
        input: text, model, tokenizer, device (CPU or GPU)
        output: the detected language is one of the following:

    """
    model.eval()
    # encode text to get the input ids and attention mask
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
            # predict the language of the text
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # select the most probable class
            _, preds = torch.max(outputs, dim=1)
    return preds

