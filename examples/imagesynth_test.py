from ezsynth import Imagesynth

style = "../styles/style000.jpg"  # 8 bit RGB
synth = Imagesynth(style)  # Create a new imagesynth object

src = "../input/000.jpg"  # 8 bit RGB
target = "../styles/style000.jpg"  # 8 bit RGB
weight = 0.5  # Weight
synth.add_guide(src, target, weight)  # Add a new guide

output = "../output/000.jpg"  # 8 bit RGB
synth.run(output)  # Run the synthesis and save.
# result = synth.run()  # Run the synthesis and return the result as a numpy array
