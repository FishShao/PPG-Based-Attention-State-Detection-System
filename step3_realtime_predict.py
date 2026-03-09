"""
step3_realtime_predict.py
Real-time PPG attention prediction + horse visualization at the end.

Usage:
    python3 step3_realtime_predict.py --model model.pkl --rounds 3
"""

import argparse, time, serial, serial.tools.list_ports as lp
import numpy as np, joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis

FS, WINDOW_SEC = 256, 30
N_SAMPLES = FS * WINDOW_SEC
TIME_LABELS = ["0 - 30s", "30s - 1min", "1min - 1min 30s"]

def bandpass(sig, fs=FS):
    b, a = butter(3, [0.5/(fs/2), 5.0/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def extract_features(seg, fs=FS):
    norm = bandpass(seg, fs)
    norm = (norm - norm.mean()) / (norm.std() + 1e-9)
    peaks, _ = find_peaks(norm, distance=int(0.4*fs), height=0.2)
    if len(peaks) < 4: return None
    ibi = np.diff(peaks) / fs * 1000
    freqs, psd = welch(norm, fs=fs, nperseg=min(256, len(norm)//2))
    lf = np.trapezoid(psd[(freqs>=0.04)&(freqs<0.15)], freqs[(freqs>=0.04)&(freqs<0.15)])
    hf = np.trapezoid(psd[(freqs>=0.15)&(freqs<0.40)], freqs[(freqs>=0.15)&(freqs<0.40)])
    return np.array([
        60000/ibi.mean(), np.std(60000/ibi),
        ibi.mean(), ibi.std(),
        np.sqrt(np.mean(np.diff(ibi)**2)), ibi.std(),
        np.sum(np.abs(np.diff(ibi))>50)/len(ibi)*100,
        lf, hf, lf/(hf+1e-9),
        len(peaks), np.mean(norm[peaks]),
        norm.std(), norm.max()-norm.min(),
        skew(norm), kurtosis(norm)
    ])

def find_port():
    for p in lp.comports():
        if any(k in p.description.lower() for k in ['usb','uart','cp210','ch340','wch']):
            return p.device
    return None

def collect(ser):
    data = []
    print(f"  Collecting {WINDOW_SEC}s - keep finger still!")
    start = time.time()
    while len(data) < N_SAMPLES:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if line and not line.startswith('#'):
                data.append(float(line))
                print(f"\r  [{int(len(data)/N_SAMPLES*100):3d}%] "
                      f"{time.time()-start:.0f}s  {len(data)}/{N_SAMPLES}", end='')
        except: pass
    print()
    return np.array(data)

def show(label, proba, rnd):
    lp, hp = proba[0]*100, proba[1]*100
    print("\n" + "="*52)
    print(f"  Round {rnd}")
    if label == 0:
        print("  STATE:  LOW Mental Workload  (relaxed)")
    else:
        print("  STATE:  HIGH Mental Workload  (stressed)")
    print(f"\n  Low  {'|'*int(lp/5)} {lp:.1f}%")
    print(f"  High {'|'*int(hp/5)} {hp:.1f}%")
    print("="*52)

def visualize_horse(results, sketchy_path, detailed_path,
                    output_path='attention_visualization.png'):
    W, H = 1200, 800
    sketchy  = Image.open(sketchy_path).convert('RGBA').resize((W, H), Image.LANCZOS)
    detailed = Image.open(detailed_path).convert('RGBA').resize((W, H), Image.LANCZOS)

    n  = len(results)
    sw = W // n

    composite = Image.new('RGBA', (W, H), (255, 255, 255, 255))
    for i, (_, label, _) in enumerate(results):
        x0, x1 = i * sw, (i + 1) * sw
        src = detailed if label == 1 else sketchy
        composite.paste(src.crop((x0, 0, x1, H)), (x0, 0))

    draw = ImageDraw.Draw(composite)
    for i in range(1, n):
        draw.line([(i * sw, 0), (i * sw, H)], fill=(60, 60, 60, 200), width=4)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#f8f8f5')
    ax.set_facecolor('#f8f8f5')
    ax.imshow(composite)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (rnd, label, proba) in enumerate(results):
        cx    = i * sw + sw // 2
        color = '#cc0000' if label == 1 else '#007700'
        state = 'HIGH Focus' if label == 1 else 'LOW Focus'
        time_label = TIME_LABELS[i] if i < len(TIME_LABELS) else f"Round {rnd}"

        ax.text(cx, 28, time_label,
                ha='center', va='top', fontsize=12, fontweight='bold', color='#222',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.85, edgecolor='#aaa', linewidth=1))

        ax.text(cx, H - 28, state,
                ha='center', va='bottom', fontsize=13, fontweight='bold', color=color,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          alpha=0.9, edgecolor=color, linewidth=2))

    ax.set_title("Attention State Visualization",
                 fontsize=15, fontweight='bold', color='#222', pad=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"Saved: {output_path}")
    plt.show()

def main(model_path, port, baud, rounds, sketchy_path, detailed_path):
    saved  = joblib.load(model_path)
    model  = saved['model']
    feats  = saved['features']
    mname  = saved.get('model_name', 'Model')
    print(f"Model loaded: {mname}  |  {len(feats)} features")

    port = port or find_port()
    if not port:
        print("ERROR: port not found. Use --port"); return
    ser = serial.Serial(port, baud, timeout=2)
    time.sleep(2); ser.reset_input_buffer()
    print(f"Connected: {port}\n")

    results = []
    for r in range(1, rounds + 1):
        print(f"\n-- Round {r}/{rounds} --------------------")
        sig = collect(ser)
        fv  = extract_features(sig)
        if fv is None:
            print("  Not enough peaks - adjust finger pressure"); continue
        label = model.predict(fv.reshape(1, -1))[0]
        proba = model.predict_proba(fv.reshape(1, -1))[0]
        results.append((r, label, proba))
        show(label, proba, r)
        if r < rounds:
            print(f"\nNext round in 5s..."); time.sleep(5)

    ser.close()

    print(f"\n-- Summary --------------------")
    for r, label, proba in results:
        state = "LOW (relaxed)" if label == 0 else "HIGH (stressed)"
        print(f"  Round {r}: {state}  Low {proba[0]*100:.1f}% / High {proba[1]*100:.1f}%")

    if results:
        print("\nGenerating horse visualization...")
        visualize_horse(results, sketchy_path, detailed_path)

    print("\nDone!")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',    default='model.pkl')
    p.add_argument('--port',     default=None)
    p.add_argument('--baud',     default=57600,               type=int)
    p.add_argument('--rounds',   default=3,                   type=int)
    p.add_argument('--sketchy',  default='sketchy_horse.png')
    p.add_argument('--detailed', default='detailed_horse.png')
    args = p.parse_args()
    main(args.model, args.port, args.baud, args.rounds,
         args.sketchy, args.detailed)