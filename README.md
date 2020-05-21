This project classifies music based on three genres - Jazz, Classical and Folk music by using keras, libroza libraries in python. Highest accuracy in the matrix shows 87.5%. 


Pre-requisite: sox, ffmpeg

Properties of each audio files after cropping – 4410Hz, 32 bits per sample, 160 kb/s bitrate,
Stereo, 30 seconds long


Feature extraction is done by MFCC (Mel-Frequency Cepstral Components)

![](images/01.png)
![](images/02.png)
![](images/03.png)
![](images/04.png)

Experiment of LSTM Recurrent Neural Network based model:

keras.optimizers.RMSprop ( rho=0.90, decay=0.00, lr=0.001 )
keras.optimizers.ADAM ( lr=0.001, rho=0.90, beta_1=0.90, beta_2=0.999, amsgrad=False )
keras.optimizers.SGD ( lr=0.001, momentum=0.00, decay=0.00, nesterov=False )
keras.optimizers.Adagrad ( lr=0.001, epsilon=None, decay=0.00 )
keras.optimizers.Adadelta ( lr=0.001, rho=0.97, epsilon=None, decay=0.00 )
keras.optimizers.Adamax ( lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00 )
keras.optimizers.Nadam ( lr=0.001, beta_1=0.90, beta_2=0.99, schedule_decay=0.004 )

Results:
![](images/05.png)



