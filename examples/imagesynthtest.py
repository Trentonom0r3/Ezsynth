from ezsynth import Imagesynth

style = "style.png"  # 8 bit RGB

src = "src.jpg"  # 8 bit RGB
target = "tgt.jpg"  # 8 bit RGB
weight = 0.5  # Weight

output = "output.png"  # 8 bit RGB

synth = Imagesynth(style)  # Create a new imagesynth object
synth.add_guide(src, target, weight)  # Add a new guide
synth.run(output)  # Run the synthesis and save.
# result = synth.run()  # Run the synthesis and return the result as a numpy array
