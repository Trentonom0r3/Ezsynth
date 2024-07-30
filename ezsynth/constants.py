EDGE_METHODS = ["PAGE", "PST", "Classic"]
DEFAULT_EDGE_METHOD = "Classic"

FLOW_MODELS = ["sintel", "kitti"]
DEFAULT_FLOW_MODEL = "sintel"

FLOW_ARCHS = ["RAFT", "EF_RAFT", "FLOW_DIFF"]
DEFAULT_FLOW_ARCH = "RAFT"

EF_RAFT_MODELS = [
    "25000_ours-sintel",
    "ours_sintel",
    "ours-things",
]
DEFAULT_EF_RAFT_MODEL = "25000_ours-sintel"

FLOW_DIFF_MODEL = "FlowDiffuser-things"
