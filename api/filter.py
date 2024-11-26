from pydub import AudioSegment
from scipy.signal import butter, lfilter, filtfilt, iirfilter
import numpy as np

def compres(audio, sample_rate):
    audio_n = np.array(audio)
    audio_n = (audio_n * 32767).astype(np.int16)
    audio_segment = AudioSegment( audio_n.tobytes(), frame_rate=sample_rate, sample_width=audio_n.dtype.itemsize, channels=1 )
    return audio_segment

def filter1(audio, sample_rate):
    # Полосовой фильтр(Баттерворта) для подавления частот выше 3000 и ниже 300 Гц и компрессия

    audio_n = audio.numpy()
    audio_n = (audio_n * 32767).astype(np.int16)
    audio_segment = AudioSegment( audio_n.tobytes(), frame_rate=sample_rate, sample_width=audio_n.dtype.itemsize, channels=1 )
    def apply_equalizer(audio, lowcut=300.0, highcut=3000.0, samplerate=48000, order=5):
        def butter_bandpass(lowcut, highcut, samplerate, order=5):
            nyquist = 0.5 * samplerate
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def bandpass_filter(data, lowcut, highcut, samplerate, order=5):
            b, a = butter_bandpass(lowcut, highcut, samplerate, order=order)
            y = lfilter(b, a, data)
            return y

        samples = np.array(audio.get_array_of_samples())
        filtered_samples = bandpass_filter(samples, lowcut, highcut, samplerate, order)
        return audio._spawn(filtered_samples.astype(np.int16))

    def apply_compression(audio, threshold=-20.0, ratio=2.0):
        compressed_audio = audio.compress_dynamic_range(threshold=threshold, ratio=ratio)
        return compressed_audio

    equalized_audio = apply_equalizer(audio_segment)
    compressed_audio = apply_compression(equalized_audio)

    return compressed_audio

def filter2(audio, sample_rate):
    audio_n = audio.numpy()
    audio_n = (audio_n * 32767).astype(np.int16)
    audio_segment = AudioSegment( audio_n.tobytes(), frame_rate=sample_rate, sample_width=audio_n.dtype.itemsize, channels=1 )

    def apply_equalizer(audio, lowcut=300.0, highcut=3000.0, samplerate=48000, order=5):
        def butter_bandpass(lowcut, highcut, samplerate, order=5):
            nyquist = 0.5 * samplerate
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def bandpass_filter(data, lowcut, highcut, samplerate, order=5):
            b, a = butter_bandpass(lowcut, highcut, samplerate, order=order)
            y = filtfilt(b, a, data)
            return y

        samples = np.array(audio.get_array_of_samples())
        filtered_samples = bandpass_filter(samples, lowcut, highcut, samplerate, order)
        return audio._spawn(filtered_samples.astype(np.int16))

    def apply_compression(audio, threshold=-20.0, ratio=2.0):
        compressed_audio = audio.compress_dynamic_range(threshold=threshold, ratio=ratio)
        return compressed_audio

    equalized_audio = apply_equalizer(audio_segment)
    compressed_audio = apply_compression(equalized_audio)

    return compressed_audio

def filter3(audio, sample_rate):
    # Тип IIR-фильтра для проектирования: bessel, тип фильтра: lowpass

    audio_n = audio.numpy()
    audio_n = (audio_n * 32767).astype(np.int16)
    audio_segment = AudioSegment( audio_n.tobytes(), frame_rate=sample_rate, sample_width=audio_n.dtype.itemsize, channels=1 )

    def apply_equalizer(audio, order=4):
        def iirfilter_f(order=4):
            b, a = iirfilter(order, Wn=0.15, rp=5, rs=60, btype='lowpass', ftype='bessel')
            return b, a

        def iir_filter(data, order=5):
            b, a = iirfilter_f(order=order)
            y = lfilter(b, a, data)
            return y

        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        filtered_samples = iir_filter(samples, order)
        return audio._spawn(filtered_samples.astype(np.int16))

    def apply_compression(audio, threshold=-20.0, ratio=2.0):
        compressed_audio = audio.compress_dynamic_range(threshold=threshold, ratio=ratio)
        return compressed_audio

    equalized_audio = apply_equalizer(audio_segment)
    compressed_audio = apply_compression(equalized_audio)

    return compressed_audio

def filter4(audio, sample_rate):
    audio_n = audio.numpy()
    audio_n = (audio_n * 32767).astype(np.int16)
    audio_segment = AudioSegment( audio_n.tobytes(), frame_rate=sample_rate, sample_width=audio_n.dtype.itemsize, channels=1 )

    def apply_equalizer(audio, order=4):
        def iirfilter_f(order=4):
            b, a = iirfilter(order, Wn=0.15, rp=5, rs=60, btype='lowpass', ftype='bessel')
            return b, a

        def iir_filter(data, order=5):
            b, a = iirfilter_f(order=order)
            y = filtfilt(b, a, data)
            return y

        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)
        filtered_samples = iir_filter(samples, order)
        return audio._spawn(filtered_samples.astype(np.int16))

    def apply_compression(audio, threshold=-20.0, ratio=2.0):
        compressed_audio = audio.compress_dynamic_range(threshold=threshold, ratio=ratio)
        return compressed_audio

    equalized_audio = apply_equalizer(audio_segment)
    compressed_audio = apply_compression(equalized_audio)

    return compressed_audio