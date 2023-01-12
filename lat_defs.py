class LatencyMetric:

    def __init__(self, flag1, flag2, flag3, flag4, flag5):
        # initializing instance variable
        self.isnidaq = flag1
        self.isframetoframe = flag2
        self.isqueue = flag3
        self.isnidaqThres = flag4
        self.isProcessTime = flag5

class LatencyData:

    def __init__(self, arr1, arr2, arr3, arr4, arr5, arr6, arr7):
        # initializing instance variable
        self.lat_nidaq = arr1
        self.lat_f2f = arr2
        self.lat_queue = arr3
        self.lat_camtrig = arr4
        self.lat_nidaq_filt = arr5
        self.lat_process_time = arr6
        self.lat_total = arr7

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