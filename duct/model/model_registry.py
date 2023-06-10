from duct.model.autoencoders import NLLVarAutoEncoder, VarAutoEncoder, AutoEncoder, \
    VQVarAutoEncoder, VQVarAutoEncoder
from duct.model.critic import Critic, MultiResCritic, SpectralCritic, MultiScaleSpectralCritic
from duct.model.patch_critic import PatchCritic1D, PatchCritic2D
from duct.model.transformer.model import Transformer, RelEmbTransformer




class ModelRegistry:
    def __init__(self):
        self.models = {
            'NLLVarAutoEncoder': NLLVarAutoEncoder,
            'Critic': Critic,
            'PatchCritic1D': PatchCritic1D,
            'PatchCritic2D': PatchCritic2D,
            'VarAutoEncoder': VarAutoEncoder,
            'AutoEncoder': AutoEncoder,
            'VQVarAutoEncoder': VQVarAutoEncoder,
            'VQVarAutoEncoder': VQVarAutoEncoder,
            'Transformer': Transformer,
            'RelEmbTransformer': RelEmbTransformer,
            'MultiResCritic': MultiResCritic,
            'SpectralCritic': SpectralCritic,
            'MultiScaleSpectralCritic': MultiScaleSpectralCritic,
        }

    def __getitem__(self, item):
        try:
            return self.models[item]
        except KeyError:
            raise KeyError(f'No model named {item} found in registry')

    def __contains__(self, item):
        return item in self.models.keys()
