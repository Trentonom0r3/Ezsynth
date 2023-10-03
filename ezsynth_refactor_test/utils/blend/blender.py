from blend.histogram_blend import HistogramBlender
from blend.reconstruction import reconstructor

class Blend():
    def __init__(self, style_fwd, style_bwd, err_masks):
        self.style_fwd = style_fwd
        self.style_bwd = style_bwd
        self.err_masks = err_masks
        self.blends = None
        pass
    
    def _hist_blend(self):
        hist_blends = []
        for i in range(len(self.style_fwd)):
            hist_blend = HistogramBlender().blend(self.style_fwd[i], self.style_bwd[i], self.err_masks[i])
            hist_blends.append(hist_blend)
        return hist_blends
    
    def _reconstruct(self, hist_blends):
        blends = reconstructor(hist_blends, self.style_fwd, self.style_bwd, self.err_masks)
        return blends
    
    def __call__(self):
        hist_blends = self._hist_blend()
        self.blends = self._reconstruct(hist_blends)
        return self.blends
    
    
