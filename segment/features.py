"""Module that implements the FeatureGenerator class."""
import numpy as np
from scipy.fftpack import dct
from .segmentaxis import segment_axis as segax


class FeatureGenerator(object):
    """
    Class that computes various speech features from an array of samples.

    Taken from speechpy, with addition of class wrapper to forgo the repeated
    calculation of filterbanks and usage of segmentaxis to stride input frames
    for a factor of 10+ increase in performance.
    """

    def __init__(self, samplerate, num_filters=40, low_freq=None, high_freq=None,
                 fft_length=512):
        """Initialize FeatureGenerator."""
        self.samplerate = samplerate
        self.num_filters = num_filters
        self.low_freq = low_freq or 300
        self.high_freq = high_freq or int(self.samplerate / 2)
        self.fft_length = fft_length
        # Calculate and store Mel filterbank.
        self.filterbank = self.calc_filterbank()

    def calc_filterbank(self):
        """Compute the Mel filterbank."""
        assert self.high_freq <= int(self.samplerate / 2)
        assert self.low_freq >= 0

        # Convert frequency from linear-domain to mel-domain.
        def freq_to_mel(x): return 1127 * np.log(1 + x / 700)

        # Convert frequency from mel-domain to linear-domain.
        def mel_to_freq(x): return 700 * (np.exp(x / 1127) - 1)

        # Compute the overlapping triangular filters used in Mel-filterbank
        # coefficient calculation.
        def triangle(x, left, middle, right):
            out = np.zeros(x.shape)
            first = np.logical_and(left < x, x <= middle)
            second = np.logical_and(middle <= x, x < right)
            out[first] = (x[first] - left) / (middle - left)
            out[second] = (right - x[second]) / (right - middle)
            return out

        mels = np.linspace(freq_to_mel(self.low_freq),
                           freq_to_mel(self.high_freq), self.num_filters + 2)
        hertz = mel_to_freq(mels)
        rfft_length = int(self.fft_length / 2) + 1
        freq_bins = (np.floor((rfft_length + 1) * hertz /
                              self.samplerate)).astype(int)
        filterbank = np.zeros((self.num_filters, rfft_length))

        for i in range(self.num_filters):
            left = freq_bins[i]
            middle = freq_bins[i + 1]
            right = freq_bins[i + 2]
            z = np.linspace(left, right, num=right - left + 1)
            filterbank[i, left:right + 1] = triangle(z, left, middle, right)

        return filterbank

    def mfe(self, signal, frame_length=0.02, frame_stride=0.01):
        """Compute Mel-filterbank coefficients and energies."""
        # Make sure samples are in a 1D ndarray of floats.
        signal = signal.astype(float)
        # Divide samples into overlapping frames of length frame_length and
        # overlap/stride frame_stride.
        frame_length = int(frame_length * self.samplerate)
        frame_stride = int(frame_stride * self.samplerate)
        frames = segax(signal, frame_length, frame_length - frame_stride,
                       end='pad', endvalue=0)
        # Compute power spectrum.
        spec = np.abs(np.fft.rfft(
            frames, n=self.fft_length, axis=-1, norm=None))
        pow_spec = 1 / self.fft_length * np.square(spec)
        # Compute frame energies.
        frame_energies = np.sum(pow_spec, axis=1)
        frame_energies[:] = np.where(frame_energies == 0, np.finfo(float).eps,
                                     frame_energies)
        # Compute Mel filterbank coefficients.
        feats = np.dot(pow_spec, self.filterbank.T)
        feats[:] = np.where(feats == 0, np.finfo(float).eps, feats)

        return feats, frame_energies

    def lmfe(self, signal, frame_length=0.02, frame_stride=0.01):
        """Compute log Mel-filterbank energies."""
        feats, _ = self.mfe(signal, frame_length, frame_stride)
        return np.log(feats)

    def mfcc(self, signal, frame_length=0.02, frame_stride=0.01,
             num_cepstrals=13, use_energy=True):
        """Compute Mel-frequency cepstral coefficients."""
        feats, energies = self.mfe(signal, frame_length, frame_stride)
        if len(feats) == 0:
            return np.empty((0, num_cepstrals))
        log_feats = np.log(feats)
        mfcc_feats = dct(log_feats, type=2, axis=-1, norm='ortho')[:,
                                                                   :num_cepstrals]
        if use_energy:
            mfcc_feats[:, 0] = np.log(energies)
        return mfcc_feats
