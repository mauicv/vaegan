from duct.model.autoencoders import NLLVarAutoEncoder, VarAutoEncoder, AutoEncoder
from duct.model.critic import Critic
from duct.model.patch_critic import NLayerDiscriminator



class ModelRegistry:
    def __init__(self):
        self.models = {
            'NLLVarAutoEncoder': NLLVarAutoEncoder,
            'Critic': Critic,
            'NLayerDiscriminator': NLayerDiscriminator,
            'VarAutoEncoder': VarAutoEncoder,
            'AutoEncoder': AutoEncoder
        }

    def __getitem__(self, item):
        try:
            return self.models[item]
        except KeyError:
            raise KeyError(f'No model named {item} found in registry')

    def __contains__(self, item):
        return item in self.models.keys()