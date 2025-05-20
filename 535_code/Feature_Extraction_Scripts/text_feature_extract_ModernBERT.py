import pickle
import json
from transformers import AutoTokenizer, AutoModel
import torch

def read_pkl_to_json(pkl_path):
    # Load the pickle file
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    # Convert non-serializable objects if needed
    def make_json_serializable(obj):
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

    # Recursively convert data
    def convert(data):
        if isinstance(data, dict):
            return {str(k): convert(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [convert(item) for item in data]
        elif isinstance(data, tuple):
            return [convert(item) for item in data]  # Convert tuple to list for JSON
        else:
            return make_json_serializable(data)

    json_ready_data = convert(data)
    return json_ready_data

pkl_path = "./dataset_extracted/language_sdk.pkl"
language_data = read_pkl_to_json(pkl_path)

modelname = 'answerdotai/ModernBERT-large'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModel.from_pretrained(modelname)
model = model.to(device)
pkl_store_path = '../../../../scratch1/jiaqilu/CSCI535/CSCI535-Project/dataset/urfunny2_text_feature_pkl/'
all_embeddings = {}
for key,data in language_data.items():
    file_id = str(key)+".pkl"
    context_sentences = data['context_sentences']
    punchline = data['punchline_sentence']
    context_one_line = "".join(context_sentences)

    context_input = tokenizer(context_one_line,return_tensors='pt') #50281 is BOS, 50282 is ESO
    punchline_input = tokenizer(punchline,return_tensors='pt') #50281 is BOS, 50282 is ESO

    #sending to gpu
    context_input = {k:v.to(device) for k,v in context_input.items()}
    punchline_input = {k:v.to(device) for k,v in punchline_input.items()}

    with torch.no_grad():
        context_output = model(**context_input)
        punchline_output = model(**punchline_input)
        context_emb = context_output.last_hidden_state[0,:,:].cpu() #raw, not including batch dimension
        punchline_emb = punchline_output.last_hidden_state[0,:,:].cpu() #raw, not including batch dimension
    pkl_name = pkl_store_path+file_id
    embeddings = {'context_embedding':context_emb,'punchline_embedding':punchline_emb}
    all_embeddings[key] = embeddings
    with open(pkl_name,'wb') as f:
        pickle.dump(embeddings,f)

pkl_whole_store_path = '../../../../scratch1/jiaqilu/CSCI535/CSCI535-Project/dataset/language_ModernBERT.pkl'
with open(pkl_whole_store_path,'wb') as pklfile:
    pickle.dump(all_embeddings,pklfile)