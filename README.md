# Speech_Recognition_Google_Speech_Commands
This repository contains the tensorflow implementation of Google Speech Commands. A total of 6 classes. Others were taken into unknown class. Those words, which were not in the google speech commands were also classifies as unknown class. Real-time words were also taken to classify them into correct class.
A test accuracy of 96% was achieved.

# Requirements:

```
tensorflow==1.14.0
librosa==0.7.0
python-speech-features==0.6
pickel==1.3
numpy==1.17.5
```
Preprocessing.py contain the feature extraction from the audio files and saving it to the pickle format
