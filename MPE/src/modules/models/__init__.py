from .rnn_agent import RNNAgent
from .target_encoder import TARGETEncoder
from .transition_network import TransitionNetwork
from .global_state_encoder import Global_State_Encoder

REGISTRY = {}

REGISTRY["rnn"] = RNNAgent
REGISTRY["target"] = TARGETEncoder
REGISTRY["transition"] = TransitionNetwork
REGISTRY["global_state"] = Global_State_Encoder