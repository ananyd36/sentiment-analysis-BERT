import config
import torch
import flask
from flask import Flask
from flask import requests
from model import BERTBaseUncased
import torch.nn as nn

app = Flask(__name__)

MODEL = None
#Device = cuda

def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_length = config.MAX_LEN
    sentence = str(sentence)
    sentence = " ".join(sentence.split())
    inputs = tokenizer.encode_plus(sentence,
        None,
        special_tokens = True,
        max_length = max_length
        )
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_length - len(ids)
    ids = ids + ([0]* padding_length)       
    mask = mask + ([0]* padding_length)
    token_type_ids = token_type_ids + ([0]* padding_length)

    
    ids = torch.tensor(ids,dtype = torch.long).unsqueeze()
    mask = torch.tensor(mask,dtype = torch.long).unsqueeze()
    token_type_ids = torch.tensor(token_type_ids,dtype = torch.long).unsqueeze()
    }

    
    outputs = model(
        ids = ids,
        mask = mask, 
        token_type_ids = token_type_ids
    )


    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

@app.route('/predict')
def predict():
    sentence =  requests.args.get("sentence")
    # print(sentence)
    positive_prediction = sentence_prediction(sentence)
    negative_prediction = 1- positive_prediction
    response = {"positive" : str(positive_prediction),
                "negative":str(negative_prediction),
                "sentence":str(sentence)
                }
    response["response"] = {}
    return flask.jsonify(response)

if __name__ = "__main__":
    model = BERTBaseUncased
    model.load_state_dict(torch.load(config.MODEL_PATH))
    #model.to(device)
    model.eval()
    app.run()    