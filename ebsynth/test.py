
from stylizevideo import ezsynth

# if __name__ == '__main__':
STYLE_PATHS = ["C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/000.jpg",
               "C:/Users/tjerf/Desktop/Testing/src/Testvids/Intermediate/099.jpg"]

IMAGE_FOLDER = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input"

OUTPUT_FOLDER = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Output"


ez = ezsynth(styles=STYLE_PATHS, imgsequence=IMAGE_FOLDER)
results = ez._set_guides().stylize(output_path=OUTPUT_FOLDER)
