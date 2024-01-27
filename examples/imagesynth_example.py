from ezsynth import Imagesynth

a = Imagesynth(style_img="input/000.jpg")

a.add_guide(
    source="input/000.jpg",
    target="styles/style000.jpg",
    weight=0.5,
)
a.run(output_path="output/000.jpg")
# or
a.run()  # run and return numpy array
