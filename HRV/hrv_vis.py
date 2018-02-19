import numpy as np
from baevskiy import SI, RRTachogrammEnergySpectrum
from buffers import Buffer

import matplotlib.pyplot as plt

__author__ = ':)'


def load_hb(fn="pulse_0.txt"):
    heart_beat_times = []
    print "[+] Reading data from :"+fn
    with open(fn, mode="r") as f:
        for l in f.readlines():
            e = l.split(";")[0]
            heart_beat_times.append(float(e))
    print "[+] "+str(len(heart_beat_times))+" heart beats was read"
    return heart_beat_times


def load_hb_fantasia(fn="Y1.txt"):
    heart_beat_times = []
    print "[+] Reading data from :"+fn
    with open(fn, mode="r") as f:
        for e in f.readlines():
            heart_beat_times.append(float(e))
    print "[+] "+str(len(heart_beat_times))+" heart beats was read"
    return heart_beat_times


def get_rrs(heart_beat_times):
    if len(heart_beat_times) < 3:
        return []
    hb = np.array(heart_beat_times)
    rrs = hb[1:] - hb[:-1]
    print "[+] " +str(len(rrs)) + " RR intervals obtained "
    return rrs




def calc_metrics(rrs):
    buf_duration_scnds=300
    tacho_buf_dur = 300
    min_rr = 0.3
    max_rr = 1.7
    sp_skip = 10
    buf_size = int(max(buf_duration_scnds, tacho_buf_dur)/min_rr)+1

    buf = Buffer(buf_size)
    print "[+] Calculating all usefull metrics: "
    si = SI(buf, buf_duration_scnds, min_rr, max_rr)
    sis = np.zeros(len(rrs))  # stress indices
    shs = np.zeros((len(si.get_histogram()), len(rrs)/sp_skip+1))  # stress histograms
    si_ready = 0

    sp = RRTachogrammEnergySpectrum(buf, tacho_buf_dur, min_rr, max_rr)
    sp_ready = 0
    sps = np.zeros((len(sp.get_spectrum()), len(rrs)/sp_skip +1))
    ics = np.zeros(len(rrs))
    iscas = np.zeros(len(rrs))
    vbs = np.zeros(len(rrs))

    hrvs = np.zeros(len(rrs))
    cnt = -1
    ls = len(rrs)
    md = ls / 10
    for r in rrs:
        if cnt % md == 0:
            print "[+] Done {0:.2f}%".format(float(100*cnt)/ls)
        cnt += 1
        si.update(r)
        sp.update(r)
        buf.add_sample(r)
        # ## Calculating stress indices
        si.calc_si()
        if si.is_ready():
            si_ready += 1
            sis[cnt] = si.get_stress()
            if cnt % sp_skip == 0:
                shs[:, cnt/sp_skip] = si.get_histogram()
        # ## Calculating RR-Tachogram spectrums
        if sp.is_ready():
            sp.calc_energy_spectrum()
            if cnt % sp_skip == 0:
                sps[:, cnt/sp_skip] = sp.get_spectrum()
            ics[cnt] = sp.get_IC()
            iscas[cnt] = sp.get_ISCA()
            vbs[cnt] = sp.get_vegetative_balance()
            sp_ready += 1
        # ## Fulfil heart rate buffer
    print "[+] Calculation finnished: SI ready " + "{0:.2f}%".format(float(si_ready*100)/cnt) + " SP ready: {0:.2f}%".format(float(sp_ready*100)/cnt)
    return sis, shs, sps, ics, iscas, vbs


def calc_hrvs(rrs, eps=0.2):
    hrvs=[]
    for r in rrs:
        if r > eps:
            hrvs.append(60.0/r)
    return hrvs


def plot_hist(shs, sps):
    plt.figure(1, figsize=(40, 10))
    plt.subplot(211)
    plt.title('Stress histograms', fontsize=14)
    plt.imshow(shs, cmap='hot', interpolation='nearest')

    plt.subplot(212)
    plt.title('RR Tachogram spectrogram', fontsize=14)
    plt.imshow(sps, cmap='hot', interpolation='nearest')
    plt.show()


def plot_si(t, sis,  ics, iscas, vbs, hrvs, rrs):
    plt.figure(1, figsize=(40, 10))

    plt.subplot(321)
    prot_hr(t, hrvs)


    plt.subplot(322)
    plt.title('RR intervals', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('RRs', fontsize=14)
    plt.plot(t[:len(rrs)], rrs)


    plt.subplot(323)
    plt.title('Stress index of time', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Stress Index', fontsize=14)
    plt.plot(t, sis)



    plt.subplot(326)
    plt.title('Index of Centralization', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('IC', fontsize=14)
    plt.plot(t, ics)


    plt.subplot(324)
    plt.title('Vegetative Balance', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Vegetative Balance', fontsize=14)
    plt.plot(t, vbs)


    plt.subplot(325)
    plt.title('Index of Subcortical Centers Activity (brain inhibition)', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('ISCA', fontsize=14)
    plt.plot(t, iscas)
    plt.show()


def prot_hr(t, hrvs, n=30):
    plt.title('Heart rate', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('beats/minute', fontsize=14)
    hr = np.convolve(hrvs, np.ones(n)/float(n), mode='vaid')
    l = min(len(t), len(hr))
    plt.plot(t[:l], hr[:l])

def plot_all(times, rrs, hrvs):
    sis, shs, sps, ics, iscas, vbs = calc_metrics(rrs)
    plot_si(times, sis, ics, iscas, vbs, hrvs, rrs)
    plot_hist(shs, sps)

list_names = ["O1", "O2", "O3", "O4", "O5", "Y1", "Y2", "Y3", "Y4", "Y5"]

if __name__ == '__main__':
    for ii in list_names:
        rrs = load_hb_fantasia(fn=ii+".txt")
        times = np.cumsum(rrs)
        hrvs = calc_hrvs(rrs)
        prot_hr(times, hrvs, 1)
        plt.show()
        plot_all(times, rrs, hrvs)