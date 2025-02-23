# Numerical-audio-authentication-system_Deep-learning
This repository implements a deep learning-based voice number authentication system using CNN and a Siamese Network. It verifies spoken numbers by comparing voice embeddings to reference samples. The model extracts audio features (MFCC, spectrogram) using CNN and determines similarity through a Siamese architecture. 

# 📌 Features
- Audio Preprocessing: Extracts MFCCs, Waveforms from numerical audio inputs.

- Siamese Network Architecture: Uses a twin neural network to compute the similarity between two audio samples.

- Triplet Loss / Contrastive Loss: Optimized for better feature embedding and verification accuracy.

- Dataset Handling: Supports labeled audio datasets with numerical recordings given by kaggle.[datset link](https://www.kaggle.com/datasets/sripaadsrinivasan/audio-mnist)

- Training & Evaluation: Implements robust training with data augmentation and real-time validation.
    - Training stat
       ![image](https://github.com/KaushiML3/Numerical-audio-authentication-system_Deep-learning/blob/main/src_img/download.png)
    - Ealuation report
    - ![image](https://github.com/KaushiML3/Numerical-audio-authentication-system_Deep-learning/blob/main/src_img/download%20(1).png))

                          precision    recall  f1-score   support
        
                real       0.92      0.94      0.93      1457
                fake       0.95      0.92      0.93      1543
        
            accuracy                           0.93      3000
           macro avg       0.93      0.93      0.93      3000
        weighted avg       0.93      0.93      0.93      3000
      
   
       

- Inference & Authentication:This uses two input audios that may or may not match the two audio files.

- Deployment Ready: Can be integrated into real-world authentication systems.
