"""
step1_extract_features.py
Extract 16 PPG features from Low_MWL/ and High_MWL/ folders

Usage:
    python3 step1_extract_features.py \
        --low_dir ./Low_MWL \
        --high_dir ./High_MWL \
        --output features_dataset.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis

FS         = 256   # sample rate (Hz)
WINDOW_SEC = 30    # window size in seconds
STEP_SEC   = 15    # step size (50% overlap)

# Bandpass filter 0.5–5 Hz
def bandpass(signal, fs=FS):
    nyq = fs / 2
    b, a = butter(3, [0.5 / nyq, 5.0 / nyq], btype='band')
    return filtfilt(b, a, signal)

#  Extract 16 features from one window 
def extract_window_features(segment, fs=FS):
    filtered = bandpass(segment, fs)
    norm = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-9)

    # Peak detection
    peaks, props = find_peaks(norm, distance=int(0.4 * fs), height=0.2)
    if len(peaks) < 4:
        return None

    ibi = np.diff(peaks) / fs * 1000  # ms

    # 1. hr_mean: mean heart rate (bpm)
    hr_mean = 60000 / np.mean(ibi)

    # 2. hr_std: std of heart rate 
    hr_std = np.std(60000 / ibi)

    # 3. ibi_mean: mean inter-beat interval (ms)
    ibi_mean = np.mean(ibi)

    # 4. ibi_std: std of IBI 
    ibi_std = np.std(ibi)

    # 5. rmssd: root mean square of successive IBI differences
    rmssd = np.sqrt(np.mean(np.diff(ibi) ** 2))

    #  6. sdnn: std of all IBI (classic HRV metric)
    sdnn = np.std(ibi)

    # 7. pnn50: % of successive IBI differences > 50ms
    pnn50 = np.sum(np.abs(np.diff(ibi)) > 50) / len(ibi) * 100

    # 8. lf_power: low frequency power (0.04–0.15 Hz)
    freqs, psd = welch(norm, fs=fs, nperseg=min(256, len(norm) // 2))
    lf_mask    = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask    = (freqs >= 0.15) & (freqs < 0.40)
    lf_power   = np.trapezoid(psd[lf_mask], freqs[lf_mask])

    # 9. hf_power: high frequency power (0.15–0.40 Hz)
    hf_power = np.trapezoid(psd[hf_mask], freqs[hf_mask])

    # 10. lf_hf_ratio: LF/HF ratio (sympathovagal balance)
    lf_hf_ratio = lf_power / (hf_power + 1e-9)

    # 11. peak_count: number of heartbeats per window
    peak_count = len(peaks)

    # 12. mean_peak_amp: mean amplitude of detected peaks
    mean_peak_amp = np.mean(norm[peaks])

    # 13. sig_std: standard deviation of normalized signal
    sig_std = np.std(norm)

    # 14. sig_range: max - min of normalized signal
    sig_range = np.max(norm) - np.min(norm)

    # 15. sig_skewness: skewness of signal distribution
    sig_skewness = skew(norm)

    # 16. sig_kurtosis: kurtosis of signal distribution
    sig_kurtosis = kurtosis(norm)

    return {
        'hr_mean':      hr_mean,
        'hr_std':       hr_std,
        'ibi_mean':     ibi_mean,
        'ibi_std':      ibi_std,
        'rmssd':        rmssd,
        'sdnn':         sdnn,
        'pnn50':        pnn50,
        'lf_power':     lf_power,
        'hf_power':     hf_power,
        'lf_hf_ratio':  lf_hf_ratio,
        'peak_count':   peak_count,
        'mean_peak_amp':mean_peak_amp,
        'sig_std':      sig_std,
        'sig_range':    sig_range,
        'sig_skewness': sig_skewness,
        'sig_kurtosis': sig_kurtosis,
    }

# Process one CSV file 
def process_file(filepath, label):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  WARNING: could not read {filepath}: {e}")
        return []

    rows = []
    win_size  = WINDOW_SEC * FS
    step_size = STEP_SEC   * FS

    for col in df.columns:
        signal = df[col].values.astype(float)
        for start in range(0, len(signal) - win_size, step_size):
            segment = signal[start : start + win_size]
            feats = extract_window_features(segment)
            if feats is not None:
                feats['label']       = label
                feats['source_file'] = os.path.basename(filepath)
                feats['column']      = col
                rows.append(feats)
    return rows

# Main
def main(low_dir, high_dir, output_file):
    all_rows = []

    print(f"\nProcessing Low MWL folder: {low_dir}")
    low_files = sorted([f for f in os.listdir(low_dir) if f.endswith('.csv')])
    print(f"Found {len(low_files)} files")
    for fname in low_files:
        rows = process_file(os.path.join(low_dir, fname), label=0)
        print(f"  {fname}: {len(rows)} windows")
        all_rows.extend(rows)

    print(f"\nProcessing High MWL folder: {high_dir}")
    high_files = sorted([f for f in os.listdir(high_dir) if f.endswith('.csv')])
    print(f"Found {len(high_files)} files")
    for fname in high_files:
        rows = process_file(os.path.join(high_dir, fname), label=1)
        print(f"  {fname}: {len(rows)} windows")
        all_rows.extend(rows)

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(output_file, index=False)

    print(f"\n{'='*50}")
    print(f"Feature extraction complete!")
    print(f"  Total samples : {len(df_out)}")
    print(f"  Low  (0)      : {(df_out.label==0).sum()} windows")
    print(f"  High (1)      : {(df_out.label==1).sum()} windows")
    print(f"  Features      : {len(df_out.columns)-3}")
    print(f"  Saved to      : {output_file}")
    print(f"{'='*50}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--low_dir',  default='./Low_MWL')
    parser.add_argument('--high_dir', default='./High_MWL')
    parser.add_argument('--output',   default='features_dataset.csv')
    args = parser.parse_args()
    main(args.low_dir, args.high_dir, args.output)