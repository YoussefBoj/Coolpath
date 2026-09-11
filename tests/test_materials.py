import numpy as np
from coolpath.materials.fusion import build_gate_lut
from coolpath.taxonomy import BFMS_NAME_TO_ID as B

def test_gating_defaults():
    lut=build_gate_lut(); assert lut[0,B['Background']]==B['Asphalt']; assert lut[8,B['Background']]==B['Foliage']; assert lut[10,B['Background']]==B['Sky']
