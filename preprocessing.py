# cores = 64, ram = 256GB, node: kottos
import mne
import pandas as pd
import os
import numpy as np
import xarray as xr
import multiprocessing as mp
from tqdm import tqdm
from config import get_preprocessing_config
import matplotlib.pyplot as plt
import time

random_seed = 42
np.random.seed(random_seed)

mne.set_log_level('WARNING')
conf = get_preprocessing_config()

INPUT_PATH = "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/raw"
OUTPUT_PATH = "/dhc/home/jannis.hajda/tuh-eeg-seizure-detection/data/preprocessed"
SAMPLING_FREQ = conf["sampling_frequency"]
CONFIGURATIONS = conf["configurations"]
CHANNELS = conf["channels"]


def extract_events_from_annotations(annotation_file):
    with open(annotation_file, "r") as f:
        annotations = f.readlines()
        events = annotations[6:]

        data = []
        for event in events:
            parts = event.split(",")

            start = float(parts[1])
            stop = float(parts[2])

            label = parts[3]
            label_map = {"bckg": 0, "seiz": 1}
            label = label_map[label]

            data.append({
                "label": label,
                "start": start,
                "stop": stop,
            })

    return pd.DataFrame(data)


def load_windows(window_length=5, overlap=2.5):
    cols = ["set", "patient_id", "session_id", "configuration", "recording_id",
            "recording_path", "event_index", "start", "stop", "label"]
    data = []

    edf_path = os.path.join(INPUT_PATH, "edf")

    for root, _, files in os.walk(edf_path):
        for file in files:
            if not file.endswith(".edf"):
                continue

            rel_path = os.path.relpath(root, edf_path)
            parts = rel_path.split("/")

            if len(parts) != 4:
                continue

            set_name, patient_id, session_id, configuration = parts

            if configuration not in CONFIGURATIONS:
                continue

            recording_path = os.path.join(root, file)
            recording_id = file.replace(".edf", "").split("_")[-1]
            annotation_path = recording_path.replace(".edf", ".csv_bi")

            if not os.path.exists(recording_path) or not os.path.exists(annotation_path):
                continue

            events = extract_events_from_annotations(annotation_path)

            for i, event in events.iterrows():
                start, stop, label = event.loc[["start", "stop", "label"]]
                label = int(label)
                duration = stop - start

                if duration < window_length:
                    continue

                while start + window_length < stop:
                    data.append({
                        "set": set_name,
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "configuration": configuration,
                        "recording_id": recording_id,
                        "recording_path": recording_path,
                        "event_index": i,
                        "start": start,
                        "stop": start + window_length,
                        "label": label,
                    })

                    start += window_length - overlap

    return pd.DataFrame(data, columns=cols)


def remove_powerline_noise(raw):
    powerline_noises = [60]

    for freq in powerline_noises:
        raw.notch_filter(freqs=freq, filter_length="auto", method="iir")

    return raw


def butterworth_filter(raw):
    iir_params = dict(order=4, ftype='butter')
    raw.filter(0.5, 50, method='iir', iir_params=iir_params)
    return raw


def crop_raw(raw, start, stop):
    """Crops the raw data based on the onset and duration, handling edge cases."""
    if stop > raw.times[-1]:
        if stop - 1 / raw.info["sfreq"] == raw.times[-1]:
            return raw.copy().crop(start, raw.times[-1], include_tmax=True), True
        else:
            return None, False
    else:
        return raw.copy().crop(start, stop, include_tmax=False), True


def convert_to_freq_domain(channel_data):
    # Number of data points
    n = len(channel_data) * 2

    # Compute FFT
    fft_result = np.fft.fft(channel_data)
    amplitude = np.abs(fft_result) / n  # Normalize magnitude

    # Create frequency axis
    frequencies = np.fft.fftfreq(n, d=1/SAMPLING_FREQ)

    # Take only the positive half of the spectrum (up to Nyquist frequency)
    half_n = n // 2
    frequencies = frequencies[:half_n]
    amplitude = amplitude[:half_n]

    return amplitude


def process_recording(recording_path, recording_windows: pd.DataFrame, domain: str = "time"):
    try:

        raw_recording = mne.io.read_raw_edf(
            recording_path, preload=True).pick(picks=CHANNELS)
        raw_recording.set_meas_date(None)

        raw_recording = remove_powerline_noise(raw_recording)
        raw_recording = butterworth_filter(raw_recording)
        raw_recording = raw_recording.resample(SAMPLING_FREQ)

        raw_windows = []

        for _, window in recording_windows.iterrows():
            patient_id, label, start, stop = window[[
                "patient_id", "label", "start", "stop"]]
            raw_window, valid = crop_raw(raw_recording, start, stop)
            if not valid:
                continue

            channel_data = raw_window.get_data()

            if domain == "freq":
                channel_data = np.apply_along_axis(
                    convert_to_freq_domain, 1, channel_data, )

            raw_windows.append({
                "patient_id": patient_id,
                "channel_data": channel_data,
                "label": label,
            })

            raw_window.close()

        raw_recording.close()

        # create xarray dataset from raw windows
        channel_data = np.stack([window["channel_data"]
                                for window in raw_windows])
        labels = np.array([window["label"] for window in raw_windows])
        patient_id = np.array([window["patient_id"] for window in raw_windows])

        data = xr.DataArray(channel_data, dims=("window", "channel", "time"), coords={
            "patient_id": ("window", patient_id),
            "label": ("window", labels),
            "channel": CHANNELS,
        })

        return data

    except Exception as e:
        print(f"Failed to process recording {recording_path}: {e}")
        return None


def process_recordings_parallel(recordings, num_processes=None, domain="time"):
    manager = mp.Manager()
    queue = manager.Queue()

    if num_processes is None:
        num_processes = mp.cpu_count()

    def listener(q, total):
        pbar = tqdm(total=total, desc="Processing recordings")
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
        print("Starting parallel processing...")
        listener_process = mp.Process(
            target=listener, args=(queue, len(recordings)))
        listener_process.start()

        data = []

        for recording_path, recording_windows in recordings:
            res = pool.apply_async(process_recording, args=(
                recording_path, recording_windows, domain), callback=callback, error_callback=error_callback)
            data.append(res)

        pool.close()
        pool.join()

        listener_process.join()

        data = [d.get() for d in data]
        data = [d for d in data if d is not None]

        print("Combining results...")

        data = xr.concat(data, dim="window")

        print("Finished processing recordings.")

        return data


if __name__ == "__main__":
    domains = ["time"]
    window_lengths = [1, 2, 5, 10, 15, 20, 25, 30]

    times_csv_path = os.path.join(OUTPUT_PATH, "times.csv")

    # Check if the file exists
    if not os.path.exists(times_csv_path):
        with open(times_csv_path, "w") as f:
            f.write("domain,window_length,overlap,time\n")
            f.close()

    for domain in domains:
        for window_length in window_lengths:
            if window_length == 1:
                overlap = 0
            else:
                overlap = window_length / 2

            print(
                f"Processing data in {domain} domain with window length {window_length} and overlap {overlap}...")

            print("Loading windows...")
            windows = load_windows(
                window_length=window_length, overlap=overlap)

            bckg_windows = windows[windows["label"] == 0]
            seiz_windows = windows[windows["label"] == 1]

            # undersample majority class
            if len(bckg_windows) > len(seiz_windows):
                bckg_windows = bckg_windows.sample(
                    n=len(seiz_windows), random_state=random_seed)
            else:
                seiz_windows = seiz_windows.sample(
                    n=len(bckg_windows), random_state=random_seed)

            windows = pd.concat([seiz_windows, bckg_windows]
                                ).reset_index(drop=True)
            print(f"Loaded {len(windows)} windows.")

            recordings = windows.groupby("recording_path")

            start_time = time.time()  # Record the start time in seconds
            preprocessed_windows = process_recordings_parallel(
                recordings, domain=domain)
            duration = time.time() - start_time

            print("Saving data to disk...")

            # write domain, window_length, overlap, start and stop time to csv file
            with open(os.path.join(OUTPUT_PATH, "times.csv"), "a") as f:
                f.write(f"{domain},{window_length},{overlap},{duration}\n")
                f.close()

            output_file = os.path.join(
                OUTPUT_PATH, f"windows_{domain}_{window_length}.nc")
            preprocessed_windows.to_netcdf(output_file)

            print(f"Saved data to {output_file}")
