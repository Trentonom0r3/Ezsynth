from examples.testclass import Test

IMAGE_FOLDER = "C:/Users/tjerf/Desktop/Testing/src/Testvids/Input"

# Initialize the Test class with the path to the image sequence directory
test = Test(imgsequence=IMAGE_FOLDER, flow_method='RAFT')

# Compute forward optical flow
forward_flow = test.compute_optical_flow(backward=False)
test.optical_flow_sequence = forward_flow

# Apply temporal smoothing
test.temporal_smoothing(window_size=3)

# Compute backward optical flow
backward_flow = test.compute_optical_flow(backward=True)

# Check flow consistency to identify occlusions
# If forward_flow and backward_flow are lists of flow fields
for i in range(len(forward_flow)):
    occlusion = test.check_flow_consistency(forward_flow[i], backward_flow[i])
    test.visualize_occlusions(occlusion, frame_index=i)




