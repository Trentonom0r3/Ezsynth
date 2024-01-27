# Ezsynth - Ebsynth Python Library

Perform things like style transfer, color transfer, inpainting, superimposition, video stylization and more!

This implementation makes use of advanced physics based edge detection and RAFT optical flow,
which leads to more accurate results during synthesis.

Ezsynth uses [Trentonom0r3/ebsynth](https://github.com/Trentonom0r3/ebsynth) - library version of Ebsynth.

# Table of Contents

- [Installation](#installation)
- [API](#api)
- [Usage](#usage)
- [FAQ](#faq)
- [TODO](#todo)
- [Contributing](#contributing)
- [Changelog](#changelog)
- [Examples](#examples)

## Installation

```sh
pip install ezsynth
```

### Build from source

#### Windows

```sh
# clone the repo
py setup.py sdist bdist_wheel
pip install ezsynth-XX.XX.XX.tar.gz
```

#### macOS & Linux

```sh
# clone the repo
git submodule update --init --recursive
(cd ebsynth && ./build-linux-cpu+cuda.sh)
# or
(cd ebsynth && ./build-macos-cpu_only.sh)
cp ebsynth/bin/ebsynth.so ezsynth/utils
```

## API

```py
class Imagesynth:
    """
    Imagesynth class for image synthesis with style transfer.

    Parameters
    ----------
    style_img : str or numpy array
        Path to the style image file or a numpy array representing the style image.
    guides : list of lists, optional
        Guides for style transfer in the format [[guide 1, guide 2, weight], ...].
    uniformity : float, optional
        Uniformity parameter for ebsynth.
    patchsize : int, optional
        Patch size parameter for ebsynth.
    pyramidlevels : int, optional
        Pyramid levels parameter for ebsynth.
    searchvoteiters : int, optional
        Search vote iterations parameter for ebsynth.
    patchmatchiters : int, optional
        Patch match iterations parameter for ebsynth.
    extrapass3x3 : bool, optional
        Extra pass 3x3 parameter for ebsynth.
    backend : str, optional
        Backend to use, default is 'cuda'.
    """

    def __init__(self, style_img, guides=[], uniformity=3500.0,
                 patchsize=5, pyramidlevels=6, searchvoteiters=12,
                 patchmatchiters=6, extrapass3x3=True, backend='cuda'):
        pass

    def add_guide(self, source, target, weight):
        """
        Add a guide to the ebsynth object.
        """
        pass

    def clear_guides(self):
        """
        Clear all guides from the ebsynth object.
        """
        pass

    def run(self, output_path=None):
        """
        Run the ebsynth process and optionally save the output.
        """
        pass
```

```py
class Ezsynth:
    """
    `Ezsynth` is the main class for the ezsynth package.

    Parameters
    ----------
    styles : list
        The paths to the style images.
    imgsequence : str
        The path to the folder containing the input images.
    edge_method : str, optional
        The method for edge detection, default is "PAGE".
        Options are "PAGE", "PST", and "Classic".
    flow_method : str, optional
        The method for optical flow computation, default is "RAFT".
        Options are "RAFT" and "DeepFlow".
    model : str, optional
        The model name for optical flow, default is "sintel".
        Options are "sintel" and "kitti".
    output_folder : str, optional
        The path to the folder where output images will be saved.
    """

    def __init__(self, styles, imgsequence, edge_method="PAGE",
                 flow_method="RAFT", model="sintel", output_folder=None):
        pass

    def run(self):
        """
        Execute the stylization process.
        """
        pass

    def save(self, base_name="output", extension=".png"):
        """
        Save the results to the specified directory.
        """
        pass
```

## Usage

### Imagesynth

```py
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
```

### Ezsynth

```py
from ezsynth import Ezsynth

a = Ezsynth(
    styles=[
        "styles/style000.jpg",
        "styles/style099.jpg",
    ],
    imgsequence="input",
    edge_method="Classic",
    flow_method="RAFT",
    model="sintel",
    output_folder="output",
)

a.run()
print(a.results)
```

## FAQ

- What is source? What is target? What does weight do?
    - Source is the unstylized version of your style frame.
        - There's many ways to use this, for example, you could make source a "cutout" of your style image and "inpaint" the target.
    - Target (in context of video stylization) is the next frame in the sequence.
        - Again. many ways to use this, as in the above example, you would make this the image you're painting the source onto.
    - Weight is the weight of the guides against the style.
        - Values over 1.0 give more weight to the guides - less style image.
        - Values under 1.0 give less weight to the guides - more style image.
    - The ebsynth Repo actually does a decent job at explaining this a bit more, with examples.

- Does this work for macOS/Linux?
    - macOS, no. Potentially in the future, but that would require contributions to the codebase from others.
    - Linux, you must compile from source.

- The Ebsynth GUI app has a mask option. How do I do that with this?
    - Currently there is no option for using masks with Ezsynth. This is currently undergoing testing.

- Does this use ebsynth.exe on the back end?
    - No, this is a custom implementation, built using a pybind11 wrapper around the original ebsynth source code.

## TODO

- Profile and optimize the process.
    - Takes much longer than I'd like in its current state. (Though it does work quite well.)

## Contributing

- Chances are, if you're willing to contribute, you're more experienced with python and programming in general than I am.
    - Though I did the vast majority of ideas behind logic and setting things up, I used GPT4 and Copilot X to write the actual code.
    - This being said, changes to the codebase are super flexible and all I require is that any changes made either match, or exceed the quality of the ground-truth sequence.

- Ground Truth Sequence:
    - Found in the `Output` Folder. These are the stylized frames you can compare with.
        - `Styles` This folder contains the keyframes used for generating the ground-truth sequence.
        - `Input` This folder contains the input image sequence used for stylization.

    - You can create new Ground-Truth Sequences, but please do so with the unaltered library, and post any and all comparisons.

- Guidelines:
    - Outside the Ground-Truth Comparison, there isn't much in terms of Guidelines.
    - Where possible, try to write Object-Oriented Code (refactoring existing code is fine, as long as the final output isn't negatively altered)
        - For public methods, detailed, numpy style doc-strings. See existing Core Classes for examples.
        - For private methods, a brief description. If the method is more than a simple utility method, an example or additional annotation would be appreciated.
        - Inline comments for things that seem unusual, or where you tried something else that didn't work vice/versa.
        - Refactoring existing code is fine, but removing functionality (unless discussed and approved) is not.
        - For changes, create a fork, make changes, and submit a PR for review.

- Main Points of work:
    - Simplification and Optimization. I Know it can be sped up, it will just take a lot of testing.
    - Thread Management, locks, optimizing blending algorithms.

- If I missed anything, or you have any questions, feel free to start a new issue or discussion.

## Changelog

- 2.1.1
    - Fixed Issues with Single style image.
- 2.1.0
    - Slight Refactorings, moved `ebsynth.py` into `_ebsynth.py` to avoid naming issues when importing module.
    - Adjusted function signature for ctypes function.
    - Kept `.dll` & `.so` named as `ebsynth`.
- [12.31.23]
    - Significant Refactoring of the Library. (In My opinion, easier to follow).
        - Various Refactorings, separations, etc to classes.
        - Utilization of `opencv` for warping vs using `torch.`
    - Improved computation time, down to ~4 minutes. (Still not great, but much better than previously ~10 minutes).
    - Implemented multithreading in `Ezsynth`, `ImageSynth` remains the same.
    - Added logic for usage of `.so` with Linux. If someone can build and contribute the `.so`, would be much appreciated.
    - Went back to `.dll` usage over `.pyd`.
        - Perhaps it was how I was setting up the `.pyd`, perhaps its was something else, but using the `.dll` and utilizing multithreading leads to huge performance gains I can't ignore, so I scrapped the `.pyd`.
    - Upload new version to `pypi`.
- [10.17.23]
    - Merge https://github.com/Trentonom0r3/Ezsynth/pull/19
    - Re-build and distribute to PyPi
- [10.10.23]
    - ezsynth.run(output_path) now results in both final images AND in-between images being saved.
- [10.03.23]
    - Linked to Ebsynth Source Code w/ Wrapper

## Examples

https://github.com/Trentonom0r3/Ezsynth/assets/130304830/2937cb02-597b-4f9b-a218-340b124dcd84
