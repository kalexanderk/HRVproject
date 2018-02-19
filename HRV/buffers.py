
__author__ = ':)'


class Buffer:
    def __init__(self, buf_len=10):
        self.len = buf_len
        self.ready = False
        self.samples = []
        self.has_new_samples = False  # is used for rare events check
        self.total_added_samples = 0

    def add_sample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.len:
            self.ready = True
            self.samples = self.samples[-self.len:]
        self.has_new_samples = True
        self.total_added_samples += 1

    def get_samples(self):
        return self.samples

    def is_ready(self):
        return self.ready

    def resize(self, new_size):
        if self.len != new_size:
            self.ready = False
        if new_size == len(self.samples):
            self.ready = True
        self.len = new_size

    def samples_were_checked(self):
        self.has_new_samples = False

    def reset(self):
        self.has_new_samples = False
        self.ready = False
        self.samples = []


class TimeSpanBuffer:
    """
    If we have buffer of delta times and we want this buffer to have dynamic size with constant time duration
    Then this class can be used.
    Usage:
    first
    buf = Buffer(K)
    time_span_tracker = TimeSpanBuffer(buf, 300) / remembers last 300 sec
    time_span_tracker.update_buf_duration(new_time_delta_to_track)
    buf.add_sample(new_time_delta_to_track)
    time_span_tracker.get_samples()

    """
    def __init__(self, buf, buf_duration_scnds=300):
        self.buf = buf      # reference for the buffer to control
        self.buf_duration = buf_duration_scnds  # desired duration
        self.buf_real_duration = 0
        self.use_N_last_RRs = 0

    def _enqueue(self, rr):
        #self.buf.add_sample(rr)
        self.buf_real_duration += rr
        self.use_N_last_RRs += 1

    def _dequeue(self, at):
        rr = self.buf.samples[at]
        self.buf_real_duration -= rr
        self.use_N_last_RRs -= 1

    def update_buf_duration(self, rr_interval):
        """
            We need just last self.buf_duration seconds in buffer. No less and possibly no more.
            So we'll calculate buffer capacity and make decisions on how many buff last elements to use.
        """
        # new RR interval will be added to buffer -> if buf is full - first rr will be erased
        if self.buf.is_ready() and self.use_N_last_RRs == len(self.buf.samples):
            self._dequeue(0)
        # determine active buf len change (move only left margin to right)[0,{1,2,3}]>> [0,1,{2,3}]
        while (self.buf_real_duration+rr_interval) > self.buf_duration and 0 < self.use_N_last_RRs <= len(self.buf.samples):
            self._dequeue(-self.use_N_last_RRs)
        # last added RR interval always gets in queue [0,1,{2}] + 3 -> [0,1,{2,3}]
        self._enqueue(rr_interval)

    def get_samples(self):
        return self.buf.get_samples()[-self.use_N_last_RRs:]

    def get_N(self):
        """
        Returns the number of last active elements in buffer that sum up to desired buf duration
        :return:
        """
        return self.use_N_last_RRs

    def get_real_duration(self):
        return self.buf_real_duration

    def get_progress(self):
        return float(self.buf_real_duration)/float(self.buf_duration)

