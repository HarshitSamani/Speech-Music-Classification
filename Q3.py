import numpy as np
from scipy.stats import multivariate_normal
from scipy.io import wavfile
from matplotlib import pyplot as plt
import os
from pathlib import Path


def load_files_from_folder(folder):
    audio = np.empty((32,1))
    for file in os.listdir(folder):
        spec = spectrogram(os.path.join(folder, file), 0.025, 0.01, 64)
        audio = np.hstack((audio, spec))
    return audio[:, 1:]


def spectrogram(audio_path: Path, window_size: float, shift: float, dft_len: int) -> np.ndarray:
    freq, signal = wavfile.read(audio_path)
    sample_size = int((len(signal) - freq*window_size)/(freq*shift) + 1)
    spec = np.zeros((int(dft_len/2),sample_size),dtype=complex)

    for i in range(sample_size):
        sample = np.fft.fft(np.hamming(400)*signal[int(i*shift*freq):int(i*shift*freq) + int(window_size*freq)], dft_len)
        spec[:,i] = sample[0:int(dft_len/2)]

    spec = np.absolute(spec)
    spec = np.log(spec)
    return spec


def K_MEANS(data, n_clusters, iter, cov_type='full'):
    # initialize first few samples as centroids
    means = data[0:n_clusters, :]
    dist = np.empty((len(data), n_clusters)) # distance matrix

    for i in range(iter):
        for k in range(n_clusters):
            dist[:, k] = np.linalg.norm(data - means[k, :], axis=1)  # euclidean distance computation

        # assign cluster corresponding to min. distance
        label = np.argmin(dist, axis=1)

        for k in range(n_clusters):
            means[k] = np.mean(data[label==k], axis=0)

    weights = np.zeros(n_clusters)
    covs = []
    for i in range(n_clusters):
        weights[i] = len(data[label==i]) / len(data)
        if cov_type == 'diag':
            covs.append(np.diag(np.diag(np.cov(data[label==i].T)))) # off-diagonal elements set to 0
        else:
            covs.append(np.cov(data[label == i].T))

    return weights, means, covs


def GMM(X, n_components, weights, means, covs, iter, cov_type):
    gamma = np.zeros((len(X), n_components))  # gamma is posterior prob. of cluster given a sample
    d = len(X)
    k = 0
    LL_iter = []  # list of Logliklihood function value at different iterations

    while k != iter:
        for i in range(n_components):
            gamma[:, i] = weights[i] * multivariate_normal.pdf(X, means[i], covs[i])

        dummy = np.sum(gamma, axis=1)  # dummy variable is total prob for each sample appearing at denominator
        LL = np.sum(np.log(dummy))
        LL_iter.append(LL)

        gamma = gamma / dummy.reshape(-1, 1) # final posterior prob computed

        for i in range(n_components):
            means[i] = np.sum(gamma[:, i].reshape(d, 1)*X, axis=0) / np.sum(gamma[:, i])
            covs[i] = np.matmul((gamma[:, i].reshape(d,1)*(X - means[i])).T, (X - means[i])) / np.sum(gamma[:, i])
            if cov_type == 'diag':
                covs[i] = np.diag(np.diag(covs[i]))
            weights[i] = np.sum(gamma[:, i]) / len(X)

        k=k+1

    plt.plot(np.arange(1, iter+1), LL_iter)
    plt.title('{} covariance, No. of components={}'.format(cov_type, n_components))
    plt.xlabel("iterations")
    plt.ylabel("Logliklihood value")
    plt.show()
    return weights, means, covs


def Gaussian_mixture_pdf(X, n_components, weights, means, covs):
    p = np.zeros(len(X))
    for i in range(n_components):
        p = p + weights[i]*multivariate_normal.pdf(X, means[i], covs[i])
    return p



train_music_folder = Path("..\A2\speech_music_classification/train/music")
train_speech_folder = Path("..\A2\speech_music_classification/train/speech")
test_folder = Path("..\A2\speech_music_classification/test")

train_music = load_files_from_folder(train_music_folder).T
train_speech = load_files_from_folder(train_speech_folder).T

test = load_files_from_folder(test_folder).T
test_label = np.zeros(48)
test_label[:24] = 1  # music datas are assigned label '1' and speech datas are assigned label '0'

# Here, I have calculated log likelihood of posterior probability of data frames of each audio and whichever log likelihood is higher audio is assigned to corresponding class.

print("diag Covariance, No. of components=2")
pred = np.zeros(len(test_label))

weights_music, means_music, covs_music = K_MEANS(train_music, 2, 5, cov_type='diag')
weights_music, means_music, covs_music = GMM(train_music, 2, weights_music, means_music, covs_music, 15, cov_type='diag')

weights_speech, means_speech, covs_speech = K_MEANS(train_speech, 2, 5, cov_type='diag')
weights_speech, means_speech, covs_speech = GMM(train_speech, 2, weights_speech, means_speech, covs_speech, 15, cov_type='diag')

a = np.log(Gaussian_mixture_pdf(test, 2, weights_music, means_music, covs_music)) # array of log of posterior prob. of each frame belong to music class
b = np.log(Gaussian_mixture_pdf(test, 2, weights_speech, means_speech, covs_speech)) # array of log of posterior prob. of each frame belong to speech class

for i in range(48):
    if np.sum(a[int(i*2998):int((i+1)*2998)]) > np.sum(b[int(i*2998):int((i+1)*2998)]): # comparison of log liklihood of each audio (summing over all frames of audio)
        pred[i] = 1
    else:
        pred[i] = 0

pred.astype(int)
print("error rate: ", np.sum(np.absolute(pred-test_label))/len(test_label)*100)


print("full Covariance, No. of components=2")
pred = np.zeros(len(test_label))

weights_music, means_music, covs_music = K_MEANS(train_music, 2, 5, cov_type='full')
weights_music, means_music, covs_music = GMM(train_music, 2, weights_music, means_music, covs_music, 25, cov_type='full')

weights_speech, means_speech, covs_speech = K_MEANS(train_speech, 2, 5, cov_type='full')
weights_speech, means_speech, covs_speech = GMM(train_speech, 2, weights_speech, means_speech, covs_speech, 25, cov_type='full')

a = np.log(Gaussian_mixture_pdf(test, 2, weights_music, means_music, covs_music))
b = np.log(Gaussian_mixture_pdf(test, 2, weights_speech, means_speech, covs_speech))

for i in range(48):
    if np.sum(a[int(i*2998):int((i+1)*2998)]) > np.sum(b[int(i*2998):int((i+1)*2998)]):
        pred[i] = 1
    else:
        pred[i] = 0

pred.astype(int)
print("error rate: ", np.sum(np.absolute(pred-test_label))/len(test_label)*100)



print("diag Covariance, No. of components=5")
pred = np.zeros(len(test_label))
weights_music, means_music, covs_music = K_MEANS(train_music, 5, 5, cov_type='diag')
weights_music, means_music, covs_music = GMM(train_music, 5, weights_music, means_music, covs_music, 30, cov_type='diag')

weights_speech, means_speech, covs_speech = K_MEANS(train_speech, 5, 5, cov_type='diag')
weights_speech, means_speech, covs_speech = GMM(train_speech, 5, weights_speech, means_speech, covs_speech, 30, cov_type='diag')

a = np.log(Gaussian_mixture_pdf(test, 5, weights_music, means_music, covs_music))
b = np.log(Gaussian_mixture_pdf(test, 5, weights_speech, means_speech, covs_speech))

for i in range(48):
    if np.sum(a[int(i*2998):int((i+1)*2998)]) > np.sum(b[int(i*2998):int((i+1)*2998)]):
        pred[i] = 1
    else:
        pred[i] = 0

pred.astype(int)
print("error rate: ", np.sum(np.absolute(pred-test_label))/len(test_label)*100)



print("full Covariance, No. of components=5")
pred = np.zeros(len(test_label))
weights_music, means_music, covs_music = K_MEANS(train_music, 5, 5, cov_type='full')
weights_music, means_music, covs_music = GMM(train_music, 5, weights_music, means_music, covs_music, 30, cov_type='full')

weights_speech, means_speech, covs_speech = K_MEANS(train_speech, 5, 5, cov_type='full')
weights_speech, means_speech, covs_speech = GMM(train_speech, 5, weights_speech, means_speech, covs_speech, 30, cov_type='full')

a = np.log(Gaussian_mixture_pdf(test, 5, weights_music, means_music, covs_music))
b = np.log(Gaussian_mixture_pdf(test, 5, weights_speech, means_speech, covs_speech))

for i in range(48):
    if np.sum(a[int(i*2998):int((i+1)*2998)]) > np.sum(b[int(i*2998):int((i+1)*2998)]):
        pred[i] = 1
    else:
        pred[i] = 0

pred.astype(int)
print("error rate: ", np.sum(np.absolute(pred-test_label))/len(test_label)*100)


# Covariance type	No. components	Classification error

# Diagonal	               2	        18.75%
# Full	                   2	        12.5%
# Diagonal	               5	        22.91%
# Full	                   5        	4.16%



