# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:52:20 2019

@author: Quentin Tedeschi
"""
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa

def graph_spectrogram(wav_file, plotpath = 'none'):
    fs = 44100
    data, rate = librosa.load(wav_file, sr=None)
    if(rate != fs):
        data=librosa.core.resample(data, rate, fs, res_type='kaiser_best', fix=True, scale=False)
    if data.size<88200:
        data=np.pad(data, (0, (88200-data.size)), 'constant')
    nfft = 200
    noverlap = 120
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
        
    plt.clf()
    return pxx, data, rate
def mic_spectrogram(data):
    fs = 44100
    nfft = 200
    noverlap = 120
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap = noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
    return pxx