import cv2

class GeneticDrawing:
    def __init__(self, img_path, seed=0, brushesRange=[[0.1, 0.3], [0.3, 0.7]]):
        self.original_img = cv2.imread(img_path)
        self.img_grey = cv2.cvtColor(self.original_img,cv2.COLOR_BGR2GRAY)
        self.img_grads = self._imgGradient(self.img_grey)
        self.myDNA = None
        self.seed = seed
        self.brushesRange = brushesRange
        self.sampling_mask = None
        
        #start with an empty black img
        self.imgBuffer = [np.zeros((self.img_grey.shape[0], self.img_grey.shape[1]), np.uint8)]

    def calcBrushRange(self, stage, total_stages):
        return [self._calcBrushSize(self.brushesRange[0], stage, total_stages), self._calcBrushSize(self.brushesRange[1], stage, total_stages)]
        
    def set_brush_range(self, ranges):
        self.brushesRange = ranges
        
    def set_sampling_mask(self, img_path):
        self.sampling_mask = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2GRAY)
        
    def create_sampling_mask(self, s, stages):
        percent = 0.2
        start_stage = int(stages*percent)
        sampling_mask = None
        if s >= start_stage:
            t = (1.0 - (s-start_stage)/max(stages-start_stage-1,1)) * 0.25 + 0.005
            sampling_mask = self.calc_sampling_mask(t)
        return sampling_mask

    def _calcBrushSize(self, brange, stage, total_stages):
        bmin = brange[0]
        bmax = brange[1]
        t = stage/max(total_stages-1, 1)
        return (bmax-bmin)*(-t*t+1)+bmin