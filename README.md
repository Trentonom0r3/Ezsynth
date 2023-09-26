# ezsynth -- Ebsynth for Python!
Ebsynth as an importable python library! 

This is quite a simple implementation. I'll be working to add more complex classes and options in the future.

Using the class provided, you can perform things like style transfer, color transfer, inpainting, superimposition, and more!
I will be providing some examples of the different use cases in coming days, for now I've included a few example scripts for a basic stylization.

# Table of Contents
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Class Definition](#class-definition)
- [Example Usage](#example-usage)
- [FAQ](#faq)
- [TODO](#todo)
- [Contributing](#contributing)

## Installation!

- Simplest way is pip installation.

```
pip install ezsynth
```

- To build from source:
    - 1. Clone the Repo.
    - 2. Download ```opencv_world480.dll```, and place it into the ```ezsynth``` folder.
    - 3. In the main parent folder, run ```py setup.py sdist bdist_wheel```
    - 4. After that has built, run ```pip install C:\Ebsynth.py\dist\ezsynth-1.2.0.1.tar.gz``` 
        - Change path as needed.
    - 5. You should have access to ezsynth and its associated classes now!

## Class Definitions:
Both classes have inline docstrings. If you use VScode, you'll see hints on usage. 

```

class Imagesynth:
    def __init__(self, style_img):
        """
        Initialize the ebsynth object for image stylization.
        
        Parameters
        ----------
        style_img: str or numpy array
        >>> str leading to file path, or numpy array
        
        guides: tuple of lists
        >>> [[guide 1, guide 2, weight], [guide 1, guide 2, weight], ...]
        >>> guide 1: str leading to file path, or numpy array
        >>> guide 2: str leading to file path, or numpy array
        >>> weight: float
        
        Example
        -------
        from ezsynth import imagesynth
        
        >>> STYLE_PATH = "Style.jpg" or np.array
        >>> SOURCE_IMG = "Source.jpg" or np.array
        >>> TARGET_IMG = "Target.jpg" or np.array
        >>> OUTPUT_PATH = "Output.jpg" or None
        
        >>> eb = imagesynth(style_img = STYLE_PATH)
        >>> eb.add_guide(source = SOURCE_IMG, target = TARGET_IMG, weight = 1.0)
        >>> eb.run(output_path = OUTPUT_PATH)
        >>> or to do something else result = eb.run()
        
        """
    
    def add_guide(self, source, target, weight):
        """
        Add a guide to the ebsynth object.
        
        Parameters
        ----------
        source: str or numpy array
        >>> str leading to file path, or numpy array
        
        target: str or numpy array
        >>> str leading to file path, or numpy array
        
        weight: float
        """
        
    def clear_guides(self):
        """
        Clears the Guides list.
        """
        self.eb.clear_guides()   

    def run(self, output_path = None):
        """
        Run ebsynth.
        
        Parameters
        ----------
        output_path: str(optional)
        >>> str leading to file path
        
        returns: numpy array
        
        """         
        return result


class Ezsynth:
    """
    Specialized subclass of ebsynth for video stylization.
    Provides methods to process sequences of images.
  
    """

    def __init__(self, styles: Union[str, List[str]], imgsequence: str,
                 flow_method: str = 'RAFT', edge_method: str = 'PAGE', flow_model: str = 'sintel'):
        """
        Initialize the Ezsynth instance for video stylization.
        
        Parameters
        ----------
        >>> styles : Union[str, List[str]]
            Path to style image(s).
            (In the form of Style1.jpg, Style2.jpg, Style01.png, Style02.png etc.)
            >>> 3-Channel, 8-bit RGB images only.
            
        >>> imgsequence : str
            Folder Path to image sequence. 
            (In the form of 0001.png, 0002.png, image01.jpg, image02.jpg, etc.)
            >>> 3-Channel, 8-bit RGB images only.
            
        >>> flow_method : str, optional
            Optical flow method, by default 'RAFT'
            >>> options: 'RAFT', 'DeepFlow'
            
        >>> edge_method : str, optional
            Edge method, by default 'PAGE'
            >>> options: 'PAGE', 'PST', 'Classic'
            
        >>> flow_model : str, optional
            Optical flow model, by default 'sintel'
            >>> options: 'sintel', 'kitti', 'things', 'chairs'

        Example
        -------
        >>> from ezsynth import Ezsynth
        
        >>> STYLE_PATHS = ["Style1.jpg", "Style2.jpg"]
        >>> IMAGE_FOLDER = "C:/Input"
        >>> OUTPUT_FOLDER = "C:/Output"
        
        >>> ez = Ezsynth(styles=STYLE_PATHS, imgsequence=IMAGE_FOLDER)
        >>> ez.set_guides().stylize(output_path=OUTPUT_FOLDER)
        >>> or to do something else results = ez.set_guides().stylize()
        
        Example (For custom Processing Options)
        ---------------------------------------
        >>> from ezsynth import Ezsynth
        
        >>> STYLE_PATHS = ["Style1.jpg", "Style2.jpg"]
        >>> IMAGE_FOLDER = "Input"
        >>> OUTPUT_FOLDER = "Output"
        
        >>> ez = Ezsynth(styles=STYLE_PATHS, imgsequence=IMAGE_FOLDER, flow_method='DeepFlow',
                        edge_method='PST')
    
        >>> ez.set_guides().stylize(output_path=OUTPUT_FOLDER)
        >>> or to do something else, results = ez.set_guides().stylize()
        """

    def set_guides(self) -> None:
        """
        Set the guides for the image sequence initialized with the Ezsynth class.
        
        Accesible Parameters
        --------------------
        >>> flow_guides : List
            Optical flow guides.
        >>> edge_guides : List
            Edge guides.
        >>> g_pos_guides : List
            Dense Correspondence guides.
            
        
        """
        return self

    def stylize(self, output_path: Optional[str] = None) -> Optional[List]:
        """
        Stylize an image sequence initialized with the Ezsynth class.

        Parameters
        ----------
        output_path : Optional[str], optional
            Path to save the stylized images, by default None

        Returns
        -------
        [list]
            List of stylized images.
        """
            return stylized_imgs


```

## Example Usage:
- For Imagesynth:
```
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
```

- For Ezsynth:
```
from ezsynth import Ezsynth


STYLE_PATHS = [
    "output000.jpg",
    "output099.jpg",
]

IMAGE_FOLDER = "C:/Input"
OUTPUT_FOLDER = "C:/Output"

ez = Ezsynth(styles=STYLE_PATHS, imgsequence=IMAGE_FOLDER, flow_model='sintel')
ez.set_guides().stylize(output_path=OUTPUT_FOLDER)
# results = ez.set_guides().stylize() # returns a list of images as numpy arrays
```

## FAQ:

- What is source? What is target? What does weight do?
    - Source is the unstylized version of your style frame. 
        - Theres many ways to use this, for example, you could make source a "cutout" of your style image and "inpaint" the target.
    - Target (in context of video stylization) is the next frame in the sequence. 
        - Again. many ways to use this, as in the above example, you would make this the image you're painting the source onto. 
    - Weight is the weight of the guides against the style. 
        - Values over 1.0 give more weight to the guides --- less style image.
        - Values under 1.0 give less weight to the guides --- more style image.
    - The ebsynth Repo actually does a decent job at explaining this a bit more, with examples. 

- Does this work for macOS/Linux?
    - macOS, no. Potentially in the future, but that would require contributions to the codebase from others.
    - Linux, soon. Just have to compile the '.pyd' properly and then double check it with the python. 

- The Ebsynth GUI app has a mask option. How do I do that with this?
    - Currently there is no option for using masks with Ezsynth. This will be updated in the future.

- Does this use ebsynth.exe on the back end?
    - No, this is a custom implementation, built using a pybind11 wrapper around the original ebsynth source code.

## TODO:
- Profile and optimize the process. 
    - Takes much longer than I'd like in its current state. (Though it does work quite well.)
