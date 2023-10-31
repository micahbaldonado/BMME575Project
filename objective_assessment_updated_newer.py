from scipy.io import wavfile
from scipy.special import expn
from scipy.fftpack import ifft
import numpy as np
import os
from scipy.io.wavfile import read
from pystoi import stoi
from logmmse import logmmse_from_file





clean_folderpath = 'x' # directory of clean files
noisy_folderpath = 'x' # directory of noisy files

cwd = os.getcwd()
output_filepath = os.environ["HOMEPATH"] 
output_filepath = os.path.join(output_filepath, "Desktop") # set as a random file in your desktop (non-permanent storage of created sound files)


cleanfiles = []
noisyfiles = []

for filename in os.listdir(clean_folderpath):
    file_path = os.path.join(clean_folderpath, filename)
    cleanfiles.append(file_path)

for filename in os.listdir(noisy_folderpath):
    file_path = os.path.join(noisy_folderpath, filename)
    noisyfiles.append(file_path)

folderlength = len(cleanfiles) # assumption: # of clean files = # of noisy files


stois = []
for i in range(folderlength):
    print(f'Processing cleanfile: {cleanfiles[i]} and noisy file: {noisyfiles[i]}')
    noisyfile = noisyfiles[i]
    cleanfile = cleanfiles[i]
    # soundFilePath should be the noisy sound file
    # output2 should be an empty directory, which is the output of logmmse

    # feed the noisy sound file in with the output2, so a denoised (from logmmse) sound file is written in the directory of output2
    
    denoised = logmmse_from_file(noisyfile)
    ## THIS IS THE NEW LOGMMSE

    # read in properties/data from the noisy and denoised sound files
    fs, noisy = read(noisyfile)
    # denosied is in same format as noisy


    # fs, denoised = read(output_filepath)

    # print(len(noisy))
    # print(len(denoised))
    noisy_length = len(noisy)
    denoised_length = len(denoised)

    
    d = stoi(noisy[0:denoised_length], denoised, fs, extended=False)
    print(d)
    # I think the best solution to the trunkation is trunkating the end of the noisy file
    print("I think this is the STOI:", stois[i])

print("The average STOI is: ", np.average(stois))



