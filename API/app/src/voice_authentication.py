from tensorflow import keras
import librosa
import pandas as pd
import numpy as np
import os

# Enable unsafe deserialization
keras.config.enable_unsafe_deserialization()
# Load the model
current_direction = os.path.dirname(os.path.abspath(__file__))
voice_authen_model = keras.models.load_model(os.path.join(current_direction,"artifact/voice_authentication_model2.keras"))


def MFcc(wav_path,duration,n_fft,window_length,hop_length,mel_bins,n_mfcc,sr=16000):
    mfccs_list=[]
    sound_sample,sr=librosa.load(wav_path,sr=16000,duration = duration)
    # Get the actual duration of the audio
    dur = librosa.get_duration(y=sound_sample, sr=sr)
    #print(dur)

    # Fix audio length if necessary
    input_length = int(sr * duration)
    if dur < duration:
      #print("Fixing audio length:",i)
      sound_sample = librosa.util.fix_length(sound_sample, size=input_length)
    
    if len(sound_sample)>=sr:
      wav1=sound_sample[0:int(sr)]
      wav2=sound_sample[int(sr*0.5):int(1.5*sr)]
      wav3=sound_sample[int(1*sr):int(2*sr)]

      for i in [wav1,wav2,wav3]:

        mfccs = librosa.feature.mfcc(y=i,n_mfcc=n_mfcc,sr=sr,n_fft=n_fft,
                                      hop_length=hop_length, win_length=window_length,
                                      n_mels = mel_bins).T
        mfccs_list.append(mfccs)
      
    else:
        mfccs = librosa.feature.mfcc(y=sound_sample,n_mfcc=n_mfcc,sr=sr,n_fft=n_fft,
                                      hop_length=hop_length, win_length=window_length,
                                      n_mels = mel_bins).T
        mfccs_list.append(mfccs)


    #print(mfccs.shape)
    #delta_mfccs = librosa.feature.delta(mfccs)
    #delta2_mfccs = librosa.feature.delta(mfccs,order=2)

    #array = np.array([[int(num)]*mfccs.shape[1]]*mfccs.shape[0])
    #print(array.shape)
    #mfccs=mfccs+((array)/100)
    #print(mfccs.shape)

    return  mfccs_list


def inference_voice_authent(cut_off,wav1_path,wav2_path):

    try:

        predict=[]


        mfccs1=MFcc(wav1_path,duration=2,n_fft=2048,window_length=1024,hop_length=512,mel_bins=60,n_mfcc=15)
        mfccs2=MFcc(wav2_path,duration=2,n_fft=2048,window_length=1024,hop_length=512,mel_bins=60,n_mfcc=15)
        #print(len(mfccs1))
        #print(len(mfccs2))

        if len(mfccs1) != len(mfccs2):
            print("Error: MFCC lists have different lengths.")

        else:
            for i,j in enumerate(mfccs1):
                for k,l in enumerate(mfccs2):
                    if k == i:
                        # Convert to NumPy arrays
                        mfcc1 = np.array(j)
                        mfcc2 = np.array(l)

                        # Debugging: Print shapes before reshaping
                        #print("Original mfccs1 shape:", mfcc1.shape)
                        #print("Original mfccs2 shape:", mfcc2.shape)

                        # Add batch and channel dimensions → (1, 87, 15, 1)
                        mfcc1 = np.expand_dims(mfcc1, axis=(0, -1))
                        mfcc2 = np.expand_dims(mfcc2, axis=(0, -1))

                        # Debugging: Print shapes after reshaping
                        #print("Reshaped mfccs1 shape:", mfcc1.shape)  # Expected: (1, 87, 15, 1)
                        #print("Reshaped mfccs2 shape:", mfcc2.shape)  # Expected: (1, 87, 15, 1)

                        # Pass as a list of inputs
                        pred = voice_authen_model.predict([mfcc1, mfcc2])
                        #print(pred)
                        predict.append(pred[0])

        print(predict)
        print(sum(predict))


        if sum(predict)>= 1:
            status=1
            return status,"voice matched"
        else:
            status=1
            return status,"voice not matched"
        
    except Exception as e:
      #print(f"An error occurred: {e}")
      status=0
      return status,e