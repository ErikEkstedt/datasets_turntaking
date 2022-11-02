# import torch
#
# import numpy.lib.stride_tricks as stride_tricks
# import numpy as np
# from numpy.fft import rfft
# from scipy.signal.windows import hann
#
#
# def nlfer_orig_pyaapt(signal, frame_size, frame_jump, nfft, parameters):
#     def stride_matrix(vector, n_lin, n_col, hop):
#         data_matrix = stride_tricks.as_strided(
#             vector,
#             shape=(n_lin, n_col),
#             strides=(vector.strides[0] * hop, vector.strides[0]),
#         )
#         return data_matrix
#
#     # ---------------------------------------------------------------
#     # Set parameters.
#     # ---------------------------------------------------------------
#     N_f0_min = np.around((parameters["f0_min"] * 2 / float(signal.new_fs)) * nfft)
#     N_f0_max = np.around((parameters["f0_max"] / float(signal.new_fs)) * nfft)
#     window = hann(frame_size + 2)[1:-1]
#     data = np.zeros((signal.size))  # Needs other array, otherwise stride and
#     data[:] = signal.filtered  # windowing will modify signal.filtered
#
#     # ---------------------------------------------------------------
#     # Main routine.
#     # ---------------------------------------------------------------
#     samples = np.arange(
#         int(np.fix(float(frame_size) / 2)),
#         signal.size - int(np.fix(float(frame_size) / 2)),
#         frame_jump,
#     )
#
#     data_matrix = np.empty((len(samples), frame_size))
#     data_matrix[:, :] = stride_matrix(data, len(samples), frame_size, frame_jump)
#     data_matrix *= window
#
#     specData = rfft(data_matrix, nfft)
#     frame_energy = np.abs(specData[:, int(N_f0_min - 1) : int(N_f0_max)]).sum(axis=1)
#     # pitch.set_energy(frame_energy, parameters['nlfer_thresh1'])
#     # pitch.set_frames_pos(samples)
#     return frame_energy, samples
#
#
# def compare():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import amfm_decompy.basic_tools as basic
#     from amfm_decompy.pYAAPT import BandpassFilter  # , yaapt
#     from datasets_turntaking.utils import load_waveform
#     import librosa
#     from librosa import display
#
#     sample_rate = 20000
#     waveform, sr = load_waveform(
#         "assets/hello.wav", sample_rate=sample_rate, normalize=True, mono=True
#     )
#
#     # WIP
#     sample_rate = 20000
#     f0_min = 60
#     f0_max = 400
#     filter_kwargs = {"order": 150, "min_hz": 50, "max_hz": 1500, "dec_factor": 1}
#     spec_kwargs = {"frame_length": 700, "hop_length": 200, "n_fft": 8192}
#     shc_kwargs = {"NH": 3, "WL": 40}
#     f0 = pyaapt(
#         waveform,
#         sample_rate,
#         f0_min,
#         f0_max,
#         filter_kwargs=filter_kwargs,
#         spec_kwargs=spec_kwargs,
#         shc_kwargs=shc_kwargs,
#     )
#
#     # ---------------------------------------------------------------
#     # Create the signal objects and filter them.
#     # ---------------------------------------------------------------
#
#     # bandpass
#     parameters = {"dec_factor": 1, "bp_forder": 150, "bp_low": 50, "bp_high": 1500}
#     # nfler
#     nlfer_params = {
#         "frame_length": 35,
#         "frame_space": 10,
#         "fft_length": 8192,
#         "f0_min": 60,
#         "f0_max": 400,
#         "NLFER_Thresh1": 0.75,
#     }
#     parameters.update(nlfer_params)
#     nfft = parameters["fft_length"]
#     frame_size = int(np.fix(parameters["frame_length"] * sample_rate / 1000))
#     frame_jump = int(np.fix(parameters["frame_space"] * sample_rate / 1000))
#
#     signal = basic.SignalObj(
#         data=waveform[0].numpy(),
#         fs=sample_rate,
#     )
#     nonlinear_sign = basic.SignalObj(signal.data**2, signal.fs)
#     fir_filter = BandpassFilter(signal.fs, parameters)
#     signal.filtered_version(fir_filter)
#     nonlinear_sign.filtered_version(fir_filter)
#
#     filtered, _ = bandpass(waveform, sample_rate)
#     nonlinear_filtered, _ = bandpass(waveform.pow(2), sample_rate)
#
#     # s = librosa.stft(signal[0, 0].numpy(), center=False)
#     # S = librosa.stft(signal[0, 1].numpy(), center=False)
#     # fig, ax = plt.subplots(2, 1, sharex=True)
#     # img = display.specshow(
#     #     librosa.amplitude_to_db(s, ref=np.max), y_axis="log", x_axis="time", ax=ax[0]
#     # )
#     # img = display.specshow(
#     #     librosa.amplitude_to_db(S, ref=np.max), y_axis="log", x_axis="time", ax=ax[1]
#     # )
#     # ax[0].set_title("Power spectrogram")
#     # fig.colorbar(img, ax=ax, format="%+2.0f dB")
#     # plt.pause(0.1)
#
#     ################################################################3
#     non_equal = (
#         (filtered.squeeze() - torch.from_numpy(signal.filtered).float()).abs() > 1e-5
#     ).sum()
#     print("bandpass error: ", non_equal)
#     non_equal = (
#         (
#             nonlinear_filtered.squeeze()
#             - torch.from_numpy(nonlinear_sign.filtered).float()
#         ).abs()
#         > 1e-5
#     ).sum()
#     print("bandpass nonlinear error: ", non_equal)
#     ################################################################3
#
#     threshold = 0.75
#     pitch_frame_energy, samples = nlfer_orig(
#         signal,
#         frame_size=frame_size,
#         frame_jump=frame_jump,
#         nfft=nfft,
#         parameters=parameters,
#     )
#     print("pitch_frame_energy: ", tuple(pitch_frame_energy.shape))
#     print("samples: ", tuple(samples.shape))
#     pitch_frame_energy = np.concatenate((np.zeros(2), pitch_frame_energy))
#     pitch_mean = pitch_frame_energy.mean()
#     pitch_frame_energy = pitch_frame_energy / pitch_mean
#     pitch_vuv = pitch_frame_energy > threshold
#
#     # NLFER
#     x_nlfer = nlfer(filtered, sample_rate, frame_size, hop_length=frame_jump)
#     vuv = x_nlfer > threshold
#     # vuv = (x_nlfer > 0.5).float()
#
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(x_nlfer[0], label="new", alpha=0.6)
#     ax.plot(vuv[0], label="vuv", alpha=0.6)
#     ax.plot(pitch_frame_energy, label="orig", alpha=0.6)
#     ax.plot(pitch_vuv, label="orig vuv", linestyle="dashed", alpha=0.6)
#     ax.legend()
#     plt.pause(0.1)
#
#     ################################################################3
#     # SHC
#     shc_ = shc(
#         filtered,
#         sample_rate,
#         n_fft=8192,
#         hop_length=frame_jump,
#         frame_size=frame_size,
#         f0_min=60,
#         f0_max=400,
#         NH=3,
#         WL=40,
#     )
#     print("shc: ", tuple(shc_.shape))
#
#     fig, ax = plt.subplots(2, 1)
#     f0_max_bin = shc_.argmax(dim=1)[0]
#     f0_max_bin[torch.logical_not(vuv[0])] = -15
#     ax[0].plot(f0_max_bin, label="shc")
#     ax[0].legend()
#     ax[1].plot(vuv[0])
#     plt.pause(0.1)
