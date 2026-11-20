# TODO: Record a mono audio
# TODO: Hann Window a large section, then find the loudest window
# TODO: Cubic spline the loud window
# TODO: Use root finding to find the max
# TODO: map to the nearest note
# TODO: Plot the note, its difference, and how sharp or flat it is
# TODO: Have a button for rereocording

from windowing import hann
import numpy as np
import matplotlib.pyplot as plt
import math
import bisect
import sounddevice as sd
import math
from matplotlib.widgets import Button

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

f = 44100
duration_record = 5
N = 16384

def harmonic_divisor(spectrum, k, low, high):
    threshold = 0.05
    peak = max(spectrum[low:high])
    for i in range(5, 1, -1):
        count = 0
        if k/i < low:
            continue
        m = []
        for j in range(1, 2*i):
            if j % i != 0:
                m.append(j)
        for j in m:
            check = int(round(j * (k/i)))
            if max(spectrum[check-1:check+2]) > threshold * peak:
                count += 1
        if count >= 0.75 * len(m):
            return i
    return 1

def my_fft(chunk):
    n = len(chunk)

    if n == 0 or n & (n - 1) != 0:
        raise ValueError("The power of the chunk was not 2.  It needs to be 2.")
   
    if n == 1:
        return [chunk[0]]

    X = [0] * n

    even = my_fft(chunk[::2])
    odd = my_fft(chunk[1::2])

    for k in range(n//2):
        w = math.e**(-2j*math.pi * k/n)
        X[k] = even[k] + w * odd[k]
        X[k + n//2] = even[k] - w * odd[k]

    return X

def interpolate(x, y, xq):
    n = len(x) - 1
    a = y

    h = [0] * n
    alpha = [0] * (n+1)
    for i in range(n):
        h[i] = x[i+1] - x[i]
    for i in range(1, n):
        alpha[i] = 3*(a[i+1] - a[i])/h[i] - 3*(a[i] - a[i-1])/h[i-1]

    mu = [0] * n
    z = [0] * n
    l = [0] * n
    l[0] = 1
    for i in range(1, n):
        l[i] = 2*(x[i+1] - x[i-1]) - h[i-1]*mu[i-1]
        mu[i] = h[i]/l[i]
        z[i] = (alpha[i] - h[i-1]*z[i-1]) / l[i]

    c = [0] * (n+1)
    b = [0] * n
    d = [0] * n
    for i in range(n-1, -1, -1):
        c[i] = z[i] - mu[i]*c[i+1]
        d[i] = (c[i+1]-c[i])/(3*h[i])
        b[i] = (a[i+1] - a[i])/h[i] - h[i]*(c[i+1] + 2*c[i])/3

    result = []
    slope = []

    for q in xq:
        j = bisect.bisect_right(x,q) - 1
        j = min(max(j, 0), n-1)
        s = q - x[j]

        result.append(a[j] + b[j]*s + c[j]*s**2 + d[j]*s**3)
        slope.append(b[j] + 2*c[j]*s + 3*d[j]*s**2)
    
    return result, slope

def bisection_method(func, x_l, x_u):
    tolerance = 1e-8
    x_m = (x_l + x_u) / 2
    if np.abs(x_u - x_l) < tolerance:
        return x_m
    if func(x_m) * func(x_l) > 0:
        return bisection_method(func, x_m, x_u)
    else:
        return bisection_method(func, x_l, x_m)

def harmonic_product_spectrum(spectrum):
    hps = spectrum.copy()
    number_of_harmonics = 5
    for h in range(2, number_of_harmonics+1):
        hps_d = spectrum[::h]
        hps[:len(hps_d)] *= hps_d
    return hps[:len(spectrum) // number_of_harmonics]

def note_check(freq):
    n = round(12 * math.log2(freq/440))
    nearest_note = 440 * 2 ** (n/12)
    cents = 1200 * math.log2(freq/nearest_note)
    MIDI = 69 + n
    octave = MIDI // 12 - 1
    name = notes[MIDI % 12]
    return name, cents, octave

def audio():
    #Record audio
    audio = sd.rec(int(duration_record * f), samplerate=f, channels=1)
    sd.wait()
    audio = audio[:,0]

    windows = list(hann(audio, N, N//4))
    loudest = max(windows, key=lambda w: np.sum(w**2))
    spectrum = np.abs(my_fft(loudest)[:N//2])

    hps = harmonic_product_spectrum(spectrum)
    low = int(60 * N/f)
    high = int(1500 * N/f)
    test_peak = max(spectrum[low:high])

    freqs = np.arange(N//2) * f / N
    k = int(np.argmax(hps[low:high])) + low
    if spectrum[k] < 0.1 * test_peak:
        k = int(np.argmax(spectrum[low:high])) + low

    x = list(freqs[k-3:k+4])
    y = list(spectrum[k-3:k+4])

    def slope(fq):
        _, result_slope = interpolate(x, y, [fq])
        return result_slope[0]

    x_l = freqs[k-1]
    x_u = freqs[k+1]

    d = harmonic_divisor(spectrum, k, low, high)
    max_value = bisection_method(slope, x_l, x_u)

    result_frequency = max_value/d
  
    return result_frequency


def main():
    header = ["Note", "Octave", "Last Frequency\nrecorded (Hz)", "Frequency\nrecorded (Hz)", "Sharp or Flat", "Off By (cents):"]
    note = "-"
    octave = "-"
    last_freq = "-"
    freq = "-"
    off_by = "-"
    cents = "-"
    values = [note, octave, last_freq, freq, off_by, cents]
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.axis("off")
    table = ax.table(cellText=[values], colLabels=header, loc="center", cellLoc="center")
    for col in range(len(header)):
        table[0, col].set_height(table[0, col].get_height() * 1.8)
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    recording = ax.text(0.5, 0.5, "Recording...", ha="center", va="center", fontsize=18, transform=ax.transAxes, visible=False)

    axbutton = plt.axes([0.7, 0.05, 0.25, 0.075])
    button = Button(
        axbutton,
        "Tune"
    )

    def update(event):
        #change display
        table.set_visible(False)
        axbutton.set_visible(False)
        recording.set_visible(True)
        fig.canvas.draw()
        fig.canvas.flush_events()

        #recording action
        last_freq_text = table[1,3].get_text().get_text()
        freq_record = audio()
        freq_text = f"{freq_record:.3f}"
        note_text, off_by_value, octave = note_check(freq_record)
        table[1, 0].get_text().set_text(str(note_text))
        table[1, 1].get_text().set_text(str(octave))
        table[1, 2].get_text().set_text(str(last_freq_text))
        table[1, 3].get_text().set_text(str(freq_text))

        off_by_text = "-"
        cents_text = "-"
        if off_by_value > 5:
            off_by_text = "Sharp"
            cents_text = f"{np.abs(off_by_value) - 5:.3f}"
        elif off_by_value < -5:
            off_by_text = "Flat"
            cents_text = f"-{np.abs(off_by_value) -5:.3f}"
        else:
            off_by_text = "Tuned"
            cents_text = "Tuned"
        table[1, 4].get_text().set_text(str(off_by_text))
        table[1, 5].get_text().set_text(str(cents_text))


        #change display
        table.set_visible(True)
        axbutton.set_visible(True)
        recording.set_visible(False)
        fig.canvas.draw_idle()

    button.on_clicked(update)

    plt.show()


if __name__ == "__main__":
    main()