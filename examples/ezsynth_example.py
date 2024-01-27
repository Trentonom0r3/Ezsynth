from ezsynth import Ezsynth

a = Ezsynth(
    styles = [
        "styles/style000.jpg",
        "styles/style099.jpg",
    ],
    imgsequence = "input",
    edge_method = "Classic",
    flow_method = "RAFT",
    model = "sintel",
    output_folder = "output",
)

a.run()
print(a.results)
