# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

from buffers import Buffer, TimeSpanBuffer


class SI:
    """
    A traditional manner of grouping cardiointervals is
    the range from 400 to 1300 ms
    with the intervals of 50 ms was constituted in perennial practice.
    Baevski HRV-Recommendations  p22/67 IInd paragraph
    Thus, 20 fixed ranges of cardio-intervals' length are pointed
    which let us to compare variable pulsograms received by different researchers
    at different groups of researches. In this case the selection capacity,
    in which the grouping and lining the variable pulsograms up, is also 5-minute standard.
    """

    def __init__(self, buf, buf_duration_scnds=300, min_rr=0.3, max_rr=1.71):
        self.time_span = TimeSpanBuffer(buf, buf_duration_scnds) # remembers last 300 sec
        self.may_calc_si = 1.0 - max_rr/buf_duration_scnds # when 90% of RR times collected in buffer - may start calculations
        # SI formula
        self.interval = 0.05
        self.hist_offset = 0.02  # percentage where significant bin starts at 2-3%
        self.min_rr = min_rr
        self.max_rr = max_rr
        self.bins = list(frange(self.min_rr, #min(list0),
                          self.max_rr + self.interval, #max(list0) + self.interval,
                          self.interval))
        self.last_hist = np.zeros(len(self.bins)-1, dtype=float)

        self.zero_eps = 0.00001
        self.AMo = 0.
        self.Mo = 0.
        self.MxDMn = 0.

        # stats
        self.min_si = 10000000
        self.max_si = 300
        self.min_progress = 1.
        self.max_progress = 0.
        self.last_si = 0.


    def _characteristics(self, rrs_lst):
        "Estimating all the necessary characteristics. Work with seconds!"
        if len(rrs_lst) > 0:
            hist, bin_edges = np.histogram(rrs_lst,
                                           self.bins,
                                           range=None,
                                           weights=None,
                                           density=None)
            hs = np.sum(hist)
            if hs < self.zero_eps:
                hs = 1
            self.last_hist = np.array(hist, dtype=float) / hs
            AMo_index = hist.tolist().index(max(hist))
            max_average = np.mean([k for k in rrs_lst if (bin_edges[AMo_index] <= k < bin_edges[AMo_index + 1])])
            bin_min = np.min(rrs_lst)
            bin_max = np.max(rrs_lst)
            # find MxDMn
            for i in xrange(len(hist)):
                if float(hist[i]) / len(rrs_lst) >= self.hist_offset:  # must be > 3%
                    bin_min = np.min([k for k in rrs_lst if (bin_edges[i] <= k < bin_edges[i + 1])])
                    break
            for i in xrange(len(hist) - 1, -1, -1):
                if float(hist[i]) / len(rrs_lst) >= self.hist_offset:  # must be 3
                    bin_max = np.max([k for k in rrs_lst if (bin_edges[i] <= k < bin_edges[i + 1])])
                    break
            #plt.hist(list0, self.bins)
            #plt.title("Histogram with bins of width %.3f secs" % self.interval)
            return float(hist.max()) / len(rrs_lst) * 100, max_average, bin_max-bin_min #in seconds
        else:
            return 0., 0., 0.

    def get_histogram(self):
        return self.last_hist

    def is_ready(self):
        return (self.Mo > self.zero_eps) and (self.MxDMn > self.zero_eps)

    def get_progress(self):
        return self.time_span.get_progress()

    def update(self, rr):
        self.time_span.update_buf_duration(rr)

    def calc_si(self):
        # when the formula can be calculated?
        p = self.get_progress()
        self.max_progress = max(self.max_progress, p)
        if p > self.may_calc_si:
            self.AMo, self.Mo, self.MxDMn = self._characteristics(self.time_span.get_samples())
        else:
            self.AMo, self.Mo, self.MxDMn = 0, 0, 0
        if self.is_ready():
            self.min_progress = min(self.min_progress, p)
            self.last_si = float(self.AMo)/(2*self.Mo*self.MxDMn) #in secs
            self.max_si = max(self.max_si, self.last_si)
            self.min_si = min(self.min_si, self.last_si)

    def get_stress(self):
        """INDEX OF REGULATORY SYSTEM TENSION
        Calculation SI- it is only one of approaches to interpretation
and estimation of the histogram (variational pulsogram). In norm SI varies within the
limits of 80-150 c.u. This parameter is very sensitive to amplification of sympathetic
tone. Small load (physical or emotional) increase SI 1,5-2 times. At significant loads
it increases 5-10 times. By illness with constant tension of regulatory systems, SI in
rest can be equal to 400-600 c.u. By coronary heart disease and with myocardial
infarction, SI in rest reaches 1000-1500 c.u.
        """
        return self.last_si

    def get_norm_si(self):
        """
        Returns from  [min_SI,max_SI] to [0,1]
        or -1 in case of few values
        """
        d = self.max_si - self.min_si
        if d > 0:
            return (self.last_si - self.min_si) / float(d)
        return -1.

    def get_znorm_si(self):
        """
        Returns from  [0,max_SI] to [0,1]
        or -1 in case of few values
        """
        if self.max_si > self.min_si:
            return self.last_si / float(self.max_si)
        return -1.


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


class RRTachogrammEnergySpectrum:
    def __init__(self, buf, buf_duration_scnds=300, min_rr=0.3, max_rr=1.71):

        self.time_span = TimeSpanBuffer(buf, buf_duration_scnds) # remembers last 300 sec
        buf_size = int(buf_duration_scnds/min_rr)+1
        # energy spectrum
        self.cum_times_buf = Buffer(buf_size)
        self.time_step = 1.0 / 67 #min_rr
        self.spectr_intervals_cnt = int(buf_duration_scnds/self.time_step) # assume 1 heart beat per second
        self.freqs = np.fft.fftfreq(self.spectr_intervals_cnt, self.time_step)
        idx = np.argsort(self.freqs)
        self.idx = [i for i in idx if 0.0 <= self.freqs[i] < 0.5]
        self.last_spectrum = np.zeros(len(self.idx), dtype=float)
        self._wnd = np.hamming(self.spectr_intervals_cnt)
        self.may_calc = 1.0 - max_rr/buf_duration_scnds # when 90% of RR times collected in buffer - may start calculations

    def _get_section_params(self, ps):
        return np.sum(ps), np.max(ps)

    def get_total_power(self):
        return self._get_section_params(self.last_spectrum)

    def get_hf_power(self):
        ids = [i for i in self.idx if 0.15 <= self.freqs[i] < 0.4]
        return self._get_section_params(self.last_spectrum[ids])

    def get_lf_power(self):
        ids = [i for i in self.idx if 0.04 <= self.freqs[i] < 0.15]
        return self._get_section_params(self.last_spectrum[ids])

    def get_vlf_power(self):
        ids = [i for i in self.idx if 0.015 <= self.freqs[i] < 0.04]
        return self._get_section_params(self.last_spectrum[ids])

    def get_ulf_power(self):
        ids = [i for i in self.idx if 0.0 <= self.freqs[i] < 0.015]
        return self._get_section_params(self.last_spectrum[ids])

    def get_IC(self):
        """
        Index of Centralization -IC (Index of centralization, IC = (VLF +LF / HF)
        IC reflects a degree of prevalence of non-respiratory sinus arrhythmia over the respiratory one.
        Actually - this is quantitative characteristic of ratio between central and independent contours
        of heart rhythm regulation.
        """
        vlf = self.get_vlf_power()[0]
        lf = self.get_lf_power()[0]
        hf = self.get_hf_power()[0]
        if hf > 0 and self.is_vlf_ready():
            return vlf + float(lf) / hf
        return 0

    def get_ISCA(self):
        """
        index of activation of sub cortical nervous centers ISCA (Index of Subcortical
        Centers Activity, ISCA = VLF / LF).

        ISCA characterizes activity of cardiovascular subcortical nervous center in relation
        to higher levels of management. The increased activity of sub cortical nervous centers
        is played by growth of ISCA.
        With help of this index processes the brain inhibiting effect can be supervised.
        """
        vlf = self.get_vlf_power()[0]
        lf = self.get_lf_power()[0]
        if lf > 0 and self.is_vlf_ready():
            return vlf / float(lf)
        return 0

    def get_vegetative_balance(self):
        """HF/LF is interpreted as a parameter of vegetative balance"""
        hf = self.get_hf_power()[0]
        lf = self.get_lf_power()[0]
        if lf > 0 and self.is_lf_ready():
            return hf / float(lf)
        return 0


    def get_freq(self):
        return self.freqs[self.idx]

    def get_spectrum(self):
        return self.last_spectrum


    def calc_energy_spectrum(self):

        #http://stackoverflow.com/questions/15382076/plotting-power-spectrum-in-python
        # get unidistant sampled times
        use_N_last_RRs = self.time_span.get_N()
        unitimes = np.linspace(self.cum_times_buf.samples[-use_N_last_RRs],
                               self.cum_times_buf.samples[-1],
                               self.spectr_intervals_cnt)

        # interpolate rr_ms_sqr
        uni_rr_ms_sqr = np.interp(unitimes,
                             self.cum_times_buf.samples[-use_N_last_RRs:],
                             #1.0/np.array(self.time_span.get_samples())) # energy levels over heart frequency
                             np.array(self.time_span.get_samples())) # 1.0/energy levels over 1.0/heart_frequency

        uni_rr_ms_sqr -= np.mean(uni_rr_ms_sqr)
        # calculate spectrum
        ps = np.abs(np.fft.fft(uni_rr_ms_sqr * self._wnd))**2
        #plt.plot(freqs[self.idx], ps[self.idx])
        #s = np.sum(ps[self.idx])
        #if s < self.zero_eps:
        #    s = 1.0
        self.last_spectrum = ps[self.idx] #/ s
        #return self.last_spectrum

    def is_hf_ready(self):

        return self.time_span.buf_real_duration > 14.

    def is_lf_ready(self):

        return self.time_span.buf_real_duration > 50.

    def is_vlf_ready(self):

        return self.time_span.buf_real_duration > 132.0

    def is_ulf_ready(self):
        """
        Different buffer duration allows to perform different calculations
        ULF        Less than 0,015      More than 66 secs
        """
        return self.time_span.buf_real_duration > 300.  # standard time

    def is_ready(self):
        return self.is_vlf_ready()
        #return self.get_progress() >= self.may_calc

    def get_progress(self):
        return self.time_span.get_progress()

    def update(self, rr):
        self.time_span.update_buf_duration(rr)
        if len(self.cum_times_buf.samples) > 0:
            self.cum_times_buf.add_sample(self.cum_times_buf.samples[-1] + rr)
        else:
            self.cum_times_buf.add_sample(rr)




