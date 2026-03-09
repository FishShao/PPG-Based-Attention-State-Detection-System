"""
capture_ppg.py
采集ESP32-C3 + 模拟PPG传感器的数据，保存为CSV并显示波形
用法: python capture_ppg.py --duration 60 --output my_ppg.csv
"""

import serial
import serial.tools.list_ports as list_ports
import time
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def find_port():
    ports = list(list_ports.comports())
    print("检测到的串口：")
    for p in ports:
        print(f"  {p.device} — {p.description}")
    for p in ports:
        desc = p.description.lower()
        if any(k in desc for k in ['usb', 'uart', 'serial', 'cp210', 'ch340', 'wch']):
            return p.device
    return None

def capture(port, baud, duration_sec, output_file):
    print(f"\n连接 {port}，波特率 {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
    except Exception as e:
        print(f"错误：无法打开串口 {port}，{e}")
        sys.exit(1)

    time.sleep(2)
    ser.reset_input_buffer()

    print(f"开始采集 {duration_sec} 秒，请把手指放在传感器上！\n")
    start = time.time()
    data = []

    while time.time() - start < duration_sec:
        try:
            line = ser.readline().decode(errors='ignore').strip()
            if not line or line.startswith('#'):
                continue
            val = float(line)
            t = time.time() - start
            data.append([t, val])
            if len(data) % 256 == 0:
                print(f"\r  {t:.0f}s / {duration_sec}s  |  {len(data)} 个样本  |  当前值: {val:.0f}", end='')
        except:
            continue

    ser.close()

    if len(data) == 0:
        print("\n没有采集到数据，检查接线和串口。")
        sys.exit(1)

    arr = np.array(data)
    np.savetxt(output_file, arr, delimiter=',',
               header='timestamp_s,ppg', comments='')

    duration = arr[-1, 0]
    fs = len(arr) / duration
    print(f"\n\n保存了 {len(arr)} 个样本 → {output_file}")
    print(f"实际采样率: {fs:.1f} Hz")

    # 画图
    plt.figure(figsize=(14, 4))
    plt.plot(arr[:, 0], arr[:, 1], lw=0.6, color='steelblue')
    plt.xlabel('时间 (秒)')
    plt.ylabel('PPG 值')
    plt.title(f'PPG 信号 — {fs:.0f} Hz，{duration:.0f} 秒')
    plt.tight_layout()
    preview = output_file.replace('.csv', '_preview.png')
    plt.savefig(preview, dpi=150)
    print(f"波形图保存至: {preview}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',     default=None,   help='串口号，如 COM3 或 /dev/ttyUSB0（不填则自动检测）')
    parser.add_argument('--baud',     default=57600,  type=int)
    parser.add_argument('--duration', default=60,     type=int, help='采集秒数')
    parser.add_argument('--output',   default='my_ppg.csv')
    args = parser.parse_args()

    port = args.port
    if port is None:
        print("自动检测串口...")
        port = find_port()
        if port is None:
            print("未找到串口，请用 --port 手动指定，例如：")
            print("  python capture_ppg.py --port /dev/ttyUSB0")
            sys.exit(1)
        print(f"使用串口: {port}")

    capture(port, args.baud, args.duration, args.output)