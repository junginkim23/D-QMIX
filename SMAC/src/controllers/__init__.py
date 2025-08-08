from .basic_controller_dqmix import BasicMAC_DQMIX
from .basic_controller import BasicMAC_QMIX

REGISTRY = {}

REGISTRY["basic_mac_dqmix"] = BasicMAC_DQMIX
REGISTRY["basic_mac"] = BasicMAC_QMIX
