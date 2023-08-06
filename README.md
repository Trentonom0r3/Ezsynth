# Ebsynth.py
Ebsynth as an importable python library! 

This is quite a simple implementation. I'll be working to add more complex classes and options in the future.

Using the class provided, you can perform things like style transfer, color transfer, inpainting, superimposition, and more!
I will be providing some examples of te different use cases in coming days, for now Ive included a few example scripts for a basic stylization.

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Notes](#notes)
- [Class Definition](#class-definition)
- [Example Usage](#example-usage)
- [Future Additions](#future-additions)
  
## Installation!

```
pip install ezsynth
```

## Notes
- Default usage is set to 'Auto'. Pass CPU to backend if you don't have CUDA.
- I compiled the .dll to be compatible with compute 6.0 GPUs and up, let me know if you have any issues please.
- Pass file paths OR numpy arrays directly!
- <Style> should be the image style you're trying to achieve.
- In the context of frame stylization, this should be the stylized version of a certain frame.
  > You could then stylize the next frame using your single style frame. A simple example is as follows:

    ```
    ebsynth = ebsynth.ebsynth(style='frame1_stylized.png', guides=[('frame1.png', 'frame2.png')])
        result_img = ebsynth.run()
    ```


## Class Definition:

```
class ebsynth:
    """
    EBSynth class provides a wrapper around the ebsynth style transfer method.

    Usage:
        ebsynth = ebsynth.ebsynth(style='style.png', guides=[('source1.png', 'target1.png')])
        result_img = ebsynth.run()
    """
    
    def __init__(self, style, guides=[], weight=None, uniformity=3500.0, 
                 patchsize=5, pyramidlevels=6, searchvoteiters=12, 
                 patchmatchiters=6, extrapass3x3=False, backend='cuda'):
        # Handling the style image
        if isinstance(style, str):
            self.style = cv2.imread(style)
        elif isinstance(style, np.ndarray):
            self.style = style
        else:
            raise ValueError("style should be either a file path or a numpy array.")

        # Handling the guide images
        self.guides = []
        for source, target in guides:
            if isinstance(source, str):
                source_img = cv2.imread(source)
            elif isinstance(source, np.ndarray):
                source_img = source
            else:
                raise ValueError("source should be either a file path or a numpy array.")

            if isinstance(target, str):
                target_img = cv2.imread(target)
            elif isinstance(target, np.ndarray):
                target_img = target
            else:
                raise ValueError("target should be either a file path or a numpy array.")

            self.guides.append((source_img, target_img))

        """
        Initialize the EBSynth wrapper.
        
        :param style: path to the style image.
        :param guides: list of tuples containing source and target guide images.
        :param weight: weights for each guide pair. Defaults to 1.0 for each pair.
        :param uniformity: uniformity weight for the style transfer.
        :param patchsize: size of the patches.
        :param pyramidlevels: number of pyramid levels.
        :param searchvoteiters: number of search/vote iterations.
        :param patchmatchiters: number of Patch-Match iterations.
        :param extrapass3x3: whether to perform an extra pass with 3x3 patches.
        :param backend: backend to use ('cpu', 'cuda', or 'auto').
        """
        
        # Store the arguments
        self.style = style
        self.guides = guides
        self.weight = weight if weight else [1.0 for _ in range(len(guides))]
        self.uniformity = uniformity
        self.patchsize = patchsize
        self.pyramidlevels = pyramidlevels
        self.searchvoteiters = searchvoteiters
        self.patchmatchiters = patchmatchiters
        self.extrapass3x3 = extrapass3x3
        
        # Define backend constants
        self.backends = {
            'cpu': EBSYNTH_BACKEND_CPU,
            'cuda': EBSYNTH_BACKEND_CUDA,
            'auto': EBSYNTH_BACKEND_AUTO
        }
        self.backend = self.backends[backend]

    def add_guide(self, source, target):
        """
        Add a new guide pair.
        
        :param source: path to the source guide image.
        :param target: path to the target guide image.
        """
        self.guides.append((source, target))
        self.weight.append(1.0)  # Default weight

    def run(self):
        """
        Run the style transfer and return the result image.
        
        :return: styled image as a numpy array.
        """
        img_style = cv2.imread(self.style)

        # Prepare the guides
        guides_processed = []
        for idx, (source, target) in enumerate(self.guides):
            source_img = cv2.imread(source)
            target_img = cv2.imread(target)
            guides_processed.append((source_img, target_img, self.weight[idx]))

        # Call the run function with the provided arguments
        result = run(img_style, guides_processed, 
                     patch_size=self.patchsize,
                     num_pyramid_levels=self.pyramidlevels,
                     num_search_vote_iters=self.searchvoteiters,
                     num_patch_match_iters=self.patchmatchiters,
                     uniformity_weight=self.uniformity,
                     extraPass3x3=self.extrapass3x3
                    )
        return result

```

## Example Usage:

> For a Single Pair of Guide Images:

```
import ebsynth
import cv2
# Define the paths to the images
style_image_path = "STYLE_IMAGE_PATH"
source_guide_path = "SOURCE_GUIDE_PATH"
target_guide_path = "TARGET_GUIDE_PATH"

# Create an instance of the EBSynth class
ebsynth = ebsynth.ebsynth(style=style_image_path, guides=[(source_guide_path, target_guide_path)])

# Run the style transfer
result_img = ebsynth.run()

# If you want to save or display the result:
cv2.imwrite("PATH_TO_YOUR_OUTPUT", result_img)
cv2.imshow("Styled Image", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

> For Multiple Pairs of Guide Images:
```
import ebsynth
import cv2

# Define the paths to the images
style_image_path = "style.png"
guide_pairs = [
    ("guide1.png", "target1.png"),
    ("guide2.png", "target2.png"),
    ("guide3.png", "target3.png"),
    ("guide4.png", "target4.png")
]

# Create an instance of the EBSynth class with multiple guide pairs
ebsynth = ebsynth.ebsynth(style=style_image_path, guides=guide_pairs)

# Run the style transfer
result_img = ebsynth.run()

# If you want to save or display the result:
cv2.imwrite("output.png", result_img)
cv2.imshow("Styled Image with Multiple Guides", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

# Future Additions

- Add more detailed documentation, examples for different use cases.
- Add some extra error handling just to be safe.
- Refactor my [stylizing-video](https://github.com/Trentonom0r3/Stylizing-Video-by-Example) project into classes, and provide the functionality with EZSYNTH (for use with video)
