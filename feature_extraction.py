import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pywt
import re
import mne
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd
import os
from config import get_feature_extraction_config
import time

mne.set_log_level('WARNING')

conf = get_feature_extraction_config()

OUTPUT_PATH = "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/features/"

SAMPLING_FREQ = conf['sampling_frequency']
CHANNELS = conf['channels']

LEFT_HEMISPHERE = conf['left_hemisphere']
RIGHT_HEMISPHERE = conf['right_hemisphere']


def calc_coeffs_features(coeffs):
    mean = np.mean(coeffs)
    median = np.median(coeffs)
    std = np.std(coeffs)
    variance = np.var(coeffs)
    skew = stats.skew(coeffs)
    kurtosis = stats.kurtosis(coeffs)
    rms = np.sqrt(np.mean(coeffs ** 2))
    energy = np.sum(coeffs ** 2)

    return {
        "mean": mean,
        "median": median,
        "variance": variance,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "rms": rms,
        "energy": energy,
    }


def extract_wavelet_features(channel_data: np.ndarray) -> dict[str, float]:
    a5, d5, d4, d3, d2, d1 = pywt.wavedec(channel_data, 'db4', level=5)

    wavelet_features = {f"{coeff}_{stat}": value
                        for coeff, data in zip(["a5", "d5", "d4", "d3"], [a5, d5, d4, d3])
                        for stat, value in calc_coeffs_features(data).items()}

    return wavelet_features


def extract_band_power(channel_data, sfreq=SAMPLING_FREQ, n_fft=256) -> dict[str, float]:
    frequency_bands = {
        "delta": (0.5, 4),
        "theta": (4, 7),
        "alpha": (7, 12),
        "beta": (12, 30),
        "gamma": (30, 50)
    }

    band_powers = {}

    n_fft = min(n_fft, sfreq)
    psds, freqs = mne.time_frequency.psd_array_welch(
        channel_data, sfreq=sfreq, n_fft=n_fft, fmin=0.5, fmax=50)

    # Calculate power within each frequency band
    for band, (fmin, fmax) in frequency_bands.items():
        # Find indices of frequencies within the band
        band_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]

        # Sum the power spectral density values within the band
        band_power = np.sum(psds[band_indices])

        band_powers[band] = band_power

    return band_powers


def calc_power_ratios(band_powers: dict[str, float]) -> dict[str, float]:
    alpha_beta_ratio = band_powers["alpha"] / \
        band_powers["beta"] if band_powers["beta"] != 0 else np.nan
    theta_beta_ratio = band_powers["theta"] / \
        band_powers["beta"] if band_powers["beta"] != 0 else np.nan
    theta_alpha_beta_ratio = (band_powers["theta"] + band_powers["alpha"]) / \
        band_powers["beta"] if band_powers["beta"] != 0 else np.nan
    theta_alpha_beta_alpha_ratio = (band_powers["theta"] + band_powers["alpha"]) / (
        band_powers["beta"] + band_powers["alpha"]) if (band_powers["beta"] + band_powers["alpha"]) != 0 else np.nan
    alpha_theta_ratio = band_powers["alpha"] / \
        band_powers["theta"] if band_powers["theta"] != 0 else np.nan
    theta_alpha_ratio = band_powers["theta"] / \
        band_powers["alpha"] if band_powers["alpha"] != 0 else np.nan

    return {
        "alpha_beta_ratio": alpha_beta_ratio,
        "theta_beta_ratio": theta_beta_ratio,
        "theta_alpha_beta_ratio": theta_alpha_beta_ratio,
        "theta_alpha_beta_alpha_ratio": theta_alpha_beta_alpha_ratio,
        "alpha_theta_ratio": alpha_theta_ratio,
        "theta_alpha_ratio": theta_alpha_ratio
    }


def calc_avg_band_powers(band_powers):
    avg_band_powers = {}
    bands = band_powers[0].keys()

    for band in bands:
        avg_band_powers[f"avg_{band}"] = np.mean(
            [channel[band] for channel in band_powers])

    return avg_band_powers


def calc_asymmetry(band_powers):
    left_power = 0
    right_power = 0

    for i, channel in enumerate(CHANNELS):
        if channel in LEFT_HEMISPHERE:
            powers = list(band_powers[i].values())
            for power in powers:
                left_power += power
        elif channel in RIGHT_HEMISPHERE:
            powers = list(band_powers[i].values())
            for power in powers:
                right_power += power

    left_power = np.log(left_power) if left_power != 0 else 0
    right_power = np.log(right_power) if right_power != 0 else 0

    asymmetry = left_power - right_power
    return asymmetry


def extract_features(window):
    patient_id = window.coords["patient_id"].values.item()
    label = window.coords["label"].values.item()

    features = {
        "patient_id": patient_id,
        "label": label
    }

    channel_band_powers = []

    for channel in CHANNELS:
        channel_data = window.sel(channel=channel).values
        wavelet_features = extract_wavelet_features(channel_data)
        band_powers = extract_band_power(channel_data)
        power_ratios = calc_power_ratios(band_powers)

        wavelet_features = {f"{channel}_{key}": value for key,
                            value in wavelet_features.items()}
        features.update(wavelet_features)

        power_ratios = {f"{channel}_{key}": value for key,
                        value in power_ratios.items()}
        features.update(power_ratios)

        channel_band_powers.append(band_powers)
        band_powers = {f"{channel}_{key}": value for key,
                       value in band_powers.items()}
        features.update(band_powers)

    asymmetry = calc_asymmetry(channel_band_powers)
    features["asymmetry"] = asymmetry

    return pd.Series(features)


def extract_features_parallel(windows, num_processes=None):
    manager = mp.Manager()
    queue = manager.Queue()

    if num_processes is None:
        num_processes = mp.cpu_count()

    def listener(q, total):
        pbar = tqdm(total=total, desc="Processing windows")
        for _ in range(total):
            q.get()
            pbar.update()
            pbar.refresh()

        pbar.close()

    def callback(_):
        queue.put(1)

    def error_callback(e):
        print(f"Error: {e}")
        queue.put(1)

    with mp.Pool(num_processes) as pool:
        print("Starting parallel feature extraction...")
        listener_process = mp.Process(
            target=listener, args=(queue, len(windows)))
        listener_process.start()

        features = []

        for window in windows:
            window_features = pool.apply_async(extract_features, args=(
                window,), callback=callback, error_callback=error_callback)
            features.append(window_features)

        pool.close()
        pool.join()

        listener_process.join()

        features = [r.get() for r in features]
        features = [r for r in features if r is not None]

        print("Combining results...")

        features = pd.concat(features, axis=1).T

        print("Finished feature extraction.")

        return features


if __name__ == "__main__":
    input_files = [
        "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_1.nc",
        "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_2.nc",
        "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_5.nc",
        "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_10.nc",
        "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_15.nc",
        "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_20.nc",
        "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_25.nc",
        "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed/windows_time_30.nc",
    ]

    times_csv_path = os.path.join(OUTPUT_PATH, "times.csv")

    # Check if the file exists
    if not os.path.exists(times_csv_path):
        with open(times_csv_path, "w") as f:
            f.write("window_length,time\n")
            f.close()

    for input_file in input_files:
        print(f"Processing {input_file}...")
        windows = xr.load_dataarray(input_file)
        window_length = windows.sizes["time"] // SAMPLING_FREQ

        start_time = time.time()
        features = extract_features_parallel(windows)
        duration = time.time() - start_time

        print(f"Saving features...")
        output_file = os.path.join(
            OUTPUT_PATH, f"features_{window_length}.csv")
        features.to_csv(f"{output_file}", index=False)

        with open(times_csv_path, "a") as f:
            f.write(f"{window_length},{duration}\n")
            f.close()

        print(f"Saved features to {output_file}.")
