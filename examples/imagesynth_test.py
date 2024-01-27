from ezsynth import Imagesynth

style = "input/000.jpg"
synth = Imagesynth(style)

src = "input/000.jpg"
target = "styles/style000.jpg"
weight = 0.5
synth.add_guide(src, target, weight)

output = "output/000.jpg"
synth.run(output)  # Run the synthesis and save.
# result = synth.run()  # Run the synthesis and return the result as a numpy array
