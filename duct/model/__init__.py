from duct.model.autoencoders import NLLVarAutoEncoder2D, VarAutoEncoder2D, AutoEncoder2D, \
    VQVarAutoEncoder2D, VQVarAutoEncoder1D
from duct.model.critic import Critic
from duct.model.patch_critic import NLayerDiscriminator



class ModelRegistry:
    def __init__(self):
        self.models = {
            'NLLVarAutoEncoder2D': NLLVarAutoEncoder2D,
            'Critic': Critic,
            'NLayerDiscriminator': NLayerDiscriminator,
            'VarAutoEncoder2D': VarAutoEncoder2D,
            'AutoEncoder2D': AutoEncoder2D,
            'VQVarAutoEncoder2D': VQVarAutoEncoder2D,
            'VQVarAutoEncoder1D': VQVarAutoEncoder1D
        }

    def __getitem__(self, item):
        try:
            return self.models[item]
        except KeyError:
            raise KeyError(f'No model named {item} found in registry')

    def __contains__(self, item):
        return item in self.models.keys()
