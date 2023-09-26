from ezsynth import Ezsynth


STYLE_PATHS = [
    "output000.jpg",
    "output099.jpg",
]

IMAGE_FOLDER = "C:/Input"
OUTPUT_FOLDER = "C:/Output"

ez = Ezsynth(styles=STYLE_PATHS, imgsequence=IMAGE_FOLDER, flow_model='sintel')
ez.set_guides().stylize(output_path=OUTPUT_FOLDER)
# results = ez.set_guides().stylize() # returns a list of images as numpy arrays

