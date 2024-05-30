from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import traceback
import torch
import safetensors.torch as st
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM
from delection_model import MultiLingualClassifier, predict_language
from translate import translate_text

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup



app = Flask(__name__)


# map each language to an index
id2lang = {3: 'English', 4:'French',13:'Spanish', 11:"Portugeese", 
8:"Italian", 12: "Russian", 14: "Sweedish", 10:"Malayalam", 2: "Dutch",
 0: "Arabic", 16:"Turkish", 5: "German", 15: "Tamil", 1: "Danish", 9: "Kannada", 6: "Greek", 7: "Hindi"}


# model id of the detection model
model_id = 'amberoad/bert-multilingual-passage-reranking-msmarco'
tokenizer = AutoTokenizer.from_pretrained(model_id)
# the device on which to run the model for inference (CPU, GPU)
device = torch.device('cpu')
num_classes = 17
# detection model
model = MultiLingualClassifier(model_id, num_classes).to(device)
state_dict = torch.load('lang_detect.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# translation model
# \Machine Learning Technical Assessment\  -- is the directory where the model is stored
model_translate = AutoModelForSeq2SeqLM.from_pretrained('\model path', use_safetensors=True)



@app.route('/detect', methods=['POST'])
def detect_language():
    

    if model:
        try:
            json_ = request.json
            # convert json to a dataframe
            query = pd.DataFrame(json_)
            preds = predict_language(query['Language'][0], model, tokenizer, device)
            language = id2lang[int(preds[0])]
            return jsonify({'Language' : language})
            
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')



@app.route('/translate', methods=['POST'])
def translate():
    

    if model_translate:
        try:
            json_ = request.json
            query = pd.DataFrame(json_)
            arabic = translate_text(query['Translation'][0], model_translate, tokenizer, device)
            
            return jsonify({'Translation' : arabic})
            
        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')



if __name__ == '__main__':
    app.run(debug=True, port=6969)
