from ezsynth import Ezsynth

style_paths = [
    "output000.jpg",
    "output099.jpg",
]

image_folder = "C:/input"
output_folder = "C:/output"

ez = Ezsynth(
    styles=style_paths,
    imgsequence=image_folder,
    edge_method="Classic",
    flow_method="RAFT",
    model='sintel',
    output_folder=output_folder,
)

ez.run()  # Run the stylization process
results = ez.results  # The results are stored in the results variable
