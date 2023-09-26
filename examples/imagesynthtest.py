from ezsynth import Imagesynth


STYLE =  "style.png" # 8 bit RGB

SRC = "src.jpg" # 8 bit RGB
TGT = "tgt.jpg" # 8 bit RGB
WGT = 0.5 # Weight 

OUTPUT = "output.png" # 8 bit RGB

synth = Imagesynth(STYLE)  # Create a new imagesynth object
synth.add_guide(SRC, TGT, WGT)  # Add a new guide
synth.run(OUTPUT)  # Run the synthesis and save.
# result = synth.run()  # Run the synthesis and return the result as a numpy array
