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