from transformers import Wav2Vec2Model,Wav2Vec2FeatureExtractor
import librosa
import os
import torch
import pickle
# for aifc issue for python 3.13+, install standard-aifc and standard-sunau
# can change to whatever variant of wav2vec2, have to check later
torch.cuda.empty_cache()
model_name = "facebook/wav2vec2-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = Wav2Vec2Model.from_pretrained(model_name)

input_dir = './dataset/urfunny2_audios'
device = torch.device('cuda')

for filename in os.listdir(input_dir):
  input_audio, sample_rate = librosa.load(input_dir + '/'+filename,  sr=16000)
  with torch.no_grad():
    i= feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
    o= model(i.input_values)
  print(filename)
  #print(o.last_hidden_state.shape)
  #print(o.extract_features.shape)

  pkl_name = '../../../../scratch1/jiaqilu/CSCI535/CSCI535-Project/dataset/urfunny2_audios_feature_pkl/'+filename.split('.')[0] + '.pkl'
  #pt_name = './dataset/urfunny2_audios_pt/'+filename.split('.')[0] + '.pt'
  #torch.save(o,pt_name)
  with open(pkl_name,'wb') as f:
    pickle.dump(o.extract_features,f)

  #pt_data = torch.load(pt_name)
  #with open(pkl_name,'rb') as f:
  #  pkl_data = pickle.load(f)

  #print(pt_data.keys())
  #print(pt_data.last_hidden_state.shape)
  #print(pt_data.extract_features.shape)

  #print(pkl_data.keys())
  #print(pkl_data.last_hidden_state.shape)
  #print(pkl_data.extract_features.shape)

  #assert torch.equal(o.last_hidden_state,pkl_data.last_hidden_state)
  #assert torch.equal(o.extract_features,pkl_data.extract_features)
  