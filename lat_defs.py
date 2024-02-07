class LatencyMetric:

    def __init__(self, flag1, flag2, flag3, flag4, flag5):
        # initializing instance variable
        self.isnidaq = flag1
        self.isframetoframe = flag2
        self.isqueue = flag3
        self.isnidaqThres = flag4
        self.isProcessTime = flag5

class LatencyData:

    def __init__(self, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8):
        # initializing instance variable
        self.lat_nidaq = arr1
        self.lat_pctime_start = arr2
        self.lat_pctime_end = arr3
        self.lat_queue = arr4
        self.lat_camtrig = arr5
        self.lat_nidaq_filt = arr6
        self.lat_process_time = arr7
        self.lat_total = arr8

class BiasConfigMode:

    def __init__(self, flag1, flag2, flag3, flag4):
        # initializing instance variable
        self.isCamOnly = flag1
        self.islogging = flag2
        self.isPlugin = flag3
        self.isJaaba = flag4

class Scores:

    def __init__(self, arr1,arr2,arr3,arr4,arr5, arr6):

        self.score_ts = arr1
        self.score_side_ts = arr2
        self.score_front_ts = arr3
        self.scores= arr4
        self.frameCount = arr5
        self.view = arr6