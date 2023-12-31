from ezsynth import Ezsynth


STYLE_PATHS = [
    "output000.jpg",
    "output099.jpg",
]

IMAGE_FOLDER = "C:/Input"
OUTPUT_FOLDER = "C:/Output"

ez = Ezsynth(styles=STYLE_PATHS, imgsequence=IMAGE_FOLDER, edge_method = "Classic",
             flow_method = "RAFT", model='sintel', output_folder=OUTPUT_FOLDER) # Create an Ezsynth object

ez.run() # Run the stylization process
results = ez.results # The results are stored in the results variable

