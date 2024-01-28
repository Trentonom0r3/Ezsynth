from ezsynth import *

e = Ebsynth()

config = Config(style_image = "input/000.jpg", guides = [("input/000.jpg", "styles/style000.jpg", 0.5)])

e(config)
