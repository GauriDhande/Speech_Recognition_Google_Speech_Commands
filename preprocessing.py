import os
import librosa
from python_speech_features import mfcc
import pickle

data_path = '/home/speech_recog/speech_commands_v0.01/'

#counting all audios with all labels
all_wav = []
all_labels = []
for folder in os.listdir(data_path):
    word_path = os.path.join(data_path, folder)
    for audio_file in os.listdir(word_path):
        all_labels.append(folder)
        all_wav.append('/home/speech_recog/speech_commands_v0.01/' + folder + '/' + audio_file)

valid_audio = []
test_audio = []
#getting the validation list
with open("/home/speech_recog/validation_list.txt", "r") as f:
    for line in f:
        valid_audio.append('/home/speech_recog/speech_commands_v0.01/'+line)
   
#getting the test list
with open("/home/speech_recog/testing_list.txt", "r") as g:
    for line in g:
        test_audio.append('/home/speech_recog/speech_commands_v0.01/'+line)
    
#getting the train list
add_list = valid_audio + test_audio
t = map(lambda s: s.strip(), add_list)
train_audio = [x for x in all_wav if x not in t]

#train data
label = ['bed', 'cat', 'dog', 'one', 'stop', 'zero']
mfcc_train = []
train_labels = [] 
for wav in train_audio:
    samples, sample_rate = librosa.load(wav, sr = 16000)
    mfccs = mfcc(samples, samplerate=sample_rate, numcep=13, nfilt=26, nfft = 512)
    a = wav.split('/')[4]
    if(len(samples) == 16000):
        if a in label:
            mfcc_train.append(mfccs)
            train_labels.append(a)
        else:
            mfcc_train.append(mfccs)
            train_labels.append('unknown')
#saving the train data to pickle
with open('/home/speech_recog/mfcc_train.pkl', 'wb') as m: 
    pickle.dump(mfcc_train, m, protocol=2)
with open('/home/speech_recog/train_labels.pkl', 'wb') as n: 
    pickle.dump(train_labels, n, protocol=2)
    
#validation data
mfcc_valid = []
valid_labels = []
c = map(lambda s: s.strip(), valid_audio)
for wav in c:
    samples, sample_rate = librosa.load(wav, sr = 16000)
    mfccs = mfcc(samples, samplerate=sample_rate, numcep=13, nfilt=26, nfft = 512)
    a = wav.split('/')[4]
    if(len(samples) == 16000):
        if a in label:
            mfcc_valid.append(mfccs)
            valid_labels.append(a)
        else:
            mfcc_valid.append(mfccs)
            valid_labels.append('unknown')
        
#saving validation data to pickle
with open('/home/speech_recog/mfcc_valid.pkl', 'wb') as o: 
    pickle.dump(mfcc_valid, o, protocol=2)
with open('/home/speech_recog/valid_labels.pkl', 'wb') as p: 
    pickle.dump(valid_labels, p, protocol=2)

#test data
mfcc_test = []
test_labels = []
d = map(lambda s: s.strip(), test_audio)
for wav in d:
    samples, sample_rate = librosa.load(wav, sr = 16000)
    mfccs = mfcc(samples, samplerate=sample_rate, numcep=13, nfilt=26, nfft = 512)
    a = wav.split('/')[4]
    if(len(samples) == 16000):
        if a in label:
            mfcc_test.append(mfccs)
            test_labels.append(a)
        else:
            mfcc_test.append(mfccs)
            test_labels.append('unknown')
        
#saving the test data to pickle
with open('/home/speech_recog/mfcc_test.pkl', 'wb') as q: 
    pickle.dump(mfcc_test, q, protocol=2)
with open('/home/speech_recog/test_labels.pkl', 'wb') as r: 
    pickle.dump(test_labels, r, protocol=2)
