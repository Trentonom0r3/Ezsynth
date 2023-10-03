import time
from ebsynth import Ebsynth

class ebsynth():
    # used to run the underlying Ebsynth pipeline
    def __init__(self, style_img, uniformity = 3500.0, patch_size = 5, pyramid_levels = 6,
                 search_vote_iters = 12, patch_match_iters = 6, extra_pass_3x3 = False):
        
        self.eb = Ebsynth(style = style_img, guides = [], uniformity = uniformity,
                          patchsize = patch_size, pyramidlevels = pyramid_levels,
                          searchvoteiters = search_vote_iters, patchmatchiters = patch_match_iters,
                          extrapass3x3 = extra_pass_3x3)
        
    def add_guide(self, source, target, weight):
        self.eb.add_guide(source, target, weight)
        
    def clear_guide(self):
        self.eb.clear_guides()
         
    # run the method using Ebsynth.run [self.eb.run] will return img, nnf
    def __call__(self):
        return self.eb.run(output_nnf=True)
        
    def run(self):
        img, err = self.__call__()
        return img, err