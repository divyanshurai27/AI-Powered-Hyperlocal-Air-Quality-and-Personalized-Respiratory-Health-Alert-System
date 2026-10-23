"""Soundtrack for brag.mp4: 120 BPM, A minor -> C major, effects on the same beat grid."""
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, fftconvolve

SR = 48000
DUR = 21.0
BEAT = 0.5
N = int(SR * DUR)
rng = np.random.default_rng(7)


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def env_adsr(n, a, d, s, r, sus_len):
    a, d, r = int(a * SR), int(d * SR), int(r * SR)
    sl = max(0, int(sus_len * SR) - a - d)
    e = np.concatenate([np.linspace(0, 1, a, False), np.linspace(1, s, d, False), np.full(sl, s), np.linspace(s, 0, r)])
    return e[:n] if len(e) >= n else np.pad(e, (0, n - len(e)))


def lp(x, f, order=2):
    return sosfilt(butter(order, f, "low", fs=SR, output="sos"), x)


def hp(x, f, order=2):
    return sosfilt(butter(order, f, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x)


def add(buf, sig, t):
    i = int(round(t * SR))
    j = min(N, i + len(sig))
    if j > i:
        buf[i:j] += sig[: j - i]


tt = lambda n: np.arange(n) / SR

# chords: (start, end, root-midi, notes)
CH = [
    (0.0, 4.0, 45, [57, 60, 64]),   # Am
    (4.0, 6.0, 41, [57, 60, 65]),   # F
    (6.0, 8.0, 48, [55, 60, 64]),   # C
    (8.0, 10.0, 43, [55, 59, 62]),  # G
    (10.0, 12.0, 45, [57, 60, 64]), # Am
    (12.0, 14.0, 41, [57, 60, 65]), # F
    (14.0, 16.0, 48, [55, 60, 64]), # C
    (16.0, 18.5, 43, [55, 59, 62]), # G
    (18.5, 21.0, 48, [55, 60, 64, 67]),  # C
]

pad = np.zeros(N); bass = np.zeros(N); arp = np.zeros(N); drums = np.zeros(N); fx = np.zeros(N)

# --- pad: detuned saws, low-passed
for s, e, root, notes in CH:
    n = int((e - s + 0.6) * SR)
    t = tt(n)
    sig = np.zeros(n)
    for m in notes:
        for det in (-0.08, 0.08):
            f = hz(m + det)
            sig += 2 * ((t * f) % 1.0) - 1
    sig = lp(sig, 1400 if s >= 6 else 900) * env_adsr(n, 0.35, 0.3, 0.8, 0.6, e - s)
    add(pad, sig * 0.05, s)

# --- bass: round sine+tri on 8ths from 6s, whole notes before
for s, e, root, _ in CH:
    if s < 3:
        n = int((e - s) * SR); t = tt(n)
        add(bass, np.sin(2 * np.pi * hz(root - 12) * t) * env_adsr(n, 0.4, 0.2, 0.7, 0.4, e - s) * 0.35, s)
        continue
    step = BEAT if s < 6 or s >= 18.5 else BEAT / 2
    k = s
    while k < e - 1e-6:
        n = int(step * SR * 0.95); t = tt(n)
        f = hz(root - 12)
        tone = np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)
        accent = 1.0 if abs((k / BEAT) - round(k / BEAT)) < 1e-6 else 0.7
        add(bass, tone * env_adsr(n, 0.005, 0.08, 0.6, 0.05, step * 0.9) * 0.32 * accent, k)
        k += step
    if s >= 18.5:
        n = int((e - s) * SR); t = tt(n)
        add(bass, np.sin(2 * np.pi * hz(root - 12) * t) * np.exp(-t * 1.2) * 0.4, s)

# --- arp plucks: 16ths from 6.0 to 18.5
def pluck(f, dur=0.22, bright=3500):
    n = int(dur * SR); t = tt(n)
    sig = (2 * ((t * f) % 1.0) - 1) * 0.5 + np.sin(2 * np.pi * f * t)
    return lp(sig, bright) * np.exp(-t * 14)

for s, e, root, notes in CH:
    if s < 6 or s >= 18.5:
        continue
    pattern = notes + [notes[0] + 12, notes[1] + 12, notes[2] + 12, notes[1] + 12, notes[0] + 12]
    k, i = s, 0
    while k < e - 1e-6:
        add(arp, pluck(hz(pattern[i % len(pattern)])) * 0.07, k)
        k += BEAT / 2; i += 1

# --- drums
def kick(amp=1.0):
    n = int(0.35 * SR); t = tt(n)
    f = 45 + 90 * np.exp(-t * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    return np.sin(ph) * np.exp(-t * 9) * amp

def hat(amp=1.0):
    n = int(0.06 * SR)
    return hp(rng.standard_normal(n), 8000) * np.exp(-tt(n) * 70) * amp

def clap(amp=1.0):
    n = int(0.18 * SR)
    return bp(rng.standard_normal(n), 900, 3000) * np.exp(-tt(n) * 22) * amp

b = 0.0
while b < 18.5 - 1e-6:
    if b >= 6:
        add(drums, kick(0.55), b)
        add(drums, hat(0.05), b + BEAT / 2)
        if round(b / BEAT) % 2 == 1:
            add(drums, clap(0.10), b)
    elif b >= 3:
        if round(b / BEAT) % 2 == 0:
            add(drums, kick(0.45), b)
        add(drums, hat(0.035), b + BEAT / 2)
    else:
        add(drums, hat(0.02), b + BEAT / 2)
    b += BEAT

# --- effects, in key, on the grid
def swell(dur, lo=300, hi=4000):
    n = int(dur * SR); t = tt(n)
    sig = bp(rng.standard_normal(n), lo, hi)
    return sig * (t / dur) ** 2.2 * 0.12

def boom():
    n = int(1.2 * SR); t = tt(n)
    return (np.sin(2 * np.pi * hz(33) * t) * np.exp(-t * 3) + lp(rng.standard_normal(n), 200) * np.exp(-t * 8) * 0.3) * 0.5

def bell(m, amp, dur=1.2):
    n = int(dur * SR); t = tt(n); f = hz(m)
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2.01 * f * t) + 0.1 * np.sin(2 * np.pi * 3 * f * t)) * np.exp(-t * 4) * amp

for cut in (3.0, 6.0, 9.0, 12.0, 16.0, 18.5):
    add(fx, swell(0.5), cut - 0.5)
add(fx, boom(), 3.0)
add(fx, boom() * 0.8, 18.5)
add(fx, bell(76, 0.05), 1.0)          # Moderate pill (E5 over Am)
add(fx, bell(81, 0.06), 3.5)          # High pill (A5)
add(fx, bell(72, 0.05), 6.0); add(fx, bell(79, 0.04), 6.0)  # logo (C/G)
for i in range(6):                     # pollutant chips (G chord, ascending)
    add(fx, bell([67, 71, 74, 79, 83, 86][i], 0.018, 0.5), 10.0 + i * 0.25)
for i in range(5):                     # factor bars (C chord)
    add(fx, bell([72, 76, 79, 84, 88][i], 0.016, 0.5), 13.0 + i * 0.25)
add(fx, bell(79, 0.05), 16.0); add(fx, bell(86, 0.035), 16.25)  # alert chime (G, D)
n = int(0.03 * SR); add(fx, bp(rng.standard_normal(n), 1500, 5000) * np.exp(-tt(n) * 200) * 0.12, 17.5)  # click
add(fx, bell(84, 0.04, 2.0), 18.5); add(fx, bell(88, 0.03, 2.0), 18.5)

# --- shared space: one small hall on pad, arp and fx
irn = int(1.6 * SR)
ir = rng.standard_normal(irn) * np.exp(-tt(irn) * 3.2)
ir = lp(ir, 5000); ir /= np.sqrt(np.sum(ir ** 2))
def verb(x, wet):
    return x + wet * fftconvolve(x, ir)[:N]

mix = verb(pad, 0.5) + bass + verb(arp, 0.35) + drums + verb(fx, 0.45)
mix = hp(mix, 30)
# end fade and gentle bus compression
t = tt(N)
mix *= np.clip((DUR - t) / 1.2, 0, 1) ** 1.5
mix *= np.clip(t / 0.08, 0, 1)
mix = np.tanh(mix * 1.6) / 1.6
mix = mix / np.max(np.abs(mix)) * 0.89
st = np.stack([mix, mix], 1)
# subtle width on arp/fx via short delay on right channel
st[:, 1] = np.roll(mix, int(0.004 * SR)) * 0.4 + mix * 0.6
wavfile.write("music.wav", SR, (st * 32767).astype(np.int16))
print("ok", st.shape)
