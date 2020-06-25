import numpy as np
import librosa
import librosa.display
import math
import re
import os
import time

import matplotlib.pyplot as plt

class FeatureExtraction:

    #PATH TO ALL DATASET FILES (IN MP3 FORMAT ONLY)
    train_folder = "./data/train_database"
    val_folder = "./data/validation_database"
    test_folder = "./data/test_database"
    main_folder = "./data"

    hop_len = None
    list_genre = ['classical', 'folk', 'jazz']



    train_X_preprocessed_data = 'data_train_input.npy'
    train_Y_preprocessed_data = 'data_train_target.npy'
    dev_X_preprocessed_data = 'data_validation_input.npy'
    dev_Y_preprocessed_data = 'data_validation_target.npy'
    test_X_preprocessed_data = 'data_test_input.npy'
    test_Y_preprocessed_data = 'data_test_target.npy'

    train_X = train_Y = None
    dev_X = dev_Y = None
    test_X = test_Y = None

    def __init__(self):
        self.hop_len = 512
        self.timeseries_length_list = []

    def load_preprocess_data(self):
        self.trainfiles_list = self.path_to_audiofiles(self.train_folder)
        self.devfiles_list = self.path_to_audiofiles(self.val_folder)
        self.testfiles_list = self.path_to_audiofiles(self.test_folder)

        all_files_list = []
        all_files_list.extend(self.trainfiles_list)
        all_files_list.extend(self.devfiles_list)
        all_files_list.extend(self.testfiles_list)


        #self.precompute_min_timeseries_len(all_files_list)
        #print("[DEBUG] total number of files: " + str(len(self.timeseries_length_list)))
        train_folder = "./data/train_database"
        val_folder = "./data/validation_database"
        test_folder = "./data/test_database"

        count_train=[]
        count_val=[]
        count_test=[]

        for f in os.listdir(train_folder):
            if os.path.isfile(os.path.join(train_folder, f)):
                count_train.append(f)
        print("Number of files for Training: ",len(count_train))

        for f in os.listdir(val_folder):
            if os.path.isfile(os.path.join(val_folder, f)):
                count_val.append(f)
        print("Number of files for Validation: ",len(count_val))

        for f in os.listdir(test_folder):
            if os.path.isfile(os.path.join(test_folder, f)):
                count_test.append(f)
        print("Number of files for Testing: ",len(count_test))


        # Training set
        self.train_X, self.train_Y = self.extract_audio_features(self.trainfiles_list)
        with open(self.train_X_preprocessed_data, 'wb') as f:
            np.save(f, self.train_X)
        with open(self.train_Y_preprocessed_data, 'wb') as f:
            self.train_Y = self.one_hot(self.train_Y)
            np.save(f, self.train_Y)

        # Validation set
        self.dev_X, self.dev_Y = self.extract_audio_features(self.devfiles_list)
        with open(self.dev_X_preprocessed_data, 'wb') as f:
            np.save(f, self.dev_X)
        with open(self.dev_Y_preprocessed_data, 'wb') as f:
            self.dev_Y = self.one_hot(self.dev_Y)
            np.save(f, self.dev_Y)

        # Test set
        self.test_X, self.test_Y = self.extract_audio_features(self.testfiles_list)
        with open(self.test_X_preprocessed_data, 'wb') as f:
            np.save(f, self.test_X)
        with open(self.test_Y_preprocessed_data, 'wb') as f:
            self.test_Y = self.one_hot(self.test_Y)
            np.save(f, self.test_Y)


    def load_deserialize_data(self):

        self.train_X = np.load(self.train_X_preprocessed_data)
        self.train_Y = np.load(self.train_Y_preprocessed_data)

        self.dev_X = np.load(self.dev_X_preprocessed_data)
        self.dev_Y = np.load(self.dev_Y_preprocessed_data)

        self.test_X = np.load(self.test_X_preprocessed_data)
        self.test_Y = np.load(self.test_Y_preprocessed_data)

    def precompute_min_timeseries_len(self, list_of_audiofiles):
        for file in list_of_audiofiles:
            print("Loading..." + str(file))
            y, sr = librosa.load(file)
            self.timeseries_length_list.append(math.ceil(len(y) / self.hop_len))

    def extract_audio_features(self, list_of_audiofiles):
        start_feature_extraction = time.process_time()

        #timeseries_length = min(self.timeseries_length_list)
        timeseries_length = 128
        data = np.zeros((len(list_of_audiofiles), timeseries_length, 33), dtype=np.float64)
        target = []

        for i, file in enumerate(list_of_audiofiles):
            start_time = time.process_time()
            y, sr = librosa.load(file)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=self.hop_len, n_mfcc=13)
            spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_len)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_len)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.hop_len)

            splits = re.split('[ .]', file)
            genre = re.split('[ /]', splits[1])[3]
            target.append(genre)

            data[i, :, 0:13] = mfcc.T[0:timeseries_length, :]
            data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
            data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
            data[i, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

            print("Features Extracted from %i of Music Track %i " % (i + 1, len(list_of_audiofiles)), end="", flush=True)
            print("    Time: %.3f"% (time.process_time() - start_time),"Seconds")

        print("Total Time: ",time.process_time() - start_feature_extraction,"Seconds")
        return data, np.expand_dims(np.asarray(target), axis=1)

    def one_hot(self, Y_genre_strings):
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(self.list_genre)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = self.list_genre.index(genre_string)
            y_one_hot[i, index] = 1
        return y_one_hot

    def path_to_audiofiles(self, dir_folder):
        list_of_audio = []
        for file in os.listdir(dir_folder):
            if file.endswith(".mp3"):
                directory = "%s/%s" % (dir_folder, file)
                list_of_audio.append(directory)
        return list_of_audio
