from duct.model.autoencoders import NLLVarAutoEncoder, VarAutoEncoder, AutoEncoder, \
    VQVarAutoEncoder, VQVarAutoEncoder
from duct.model.critic import Critic
from duct.model.patch_critic import NLayerDiscriminator
from duct.model.transformer.model import Transformer, MultiScaleTransformer
from duct.model.transformer_critic import TransformerCritic



class ModelRegistry:
    def __init__(self):
        self.models = {
            'NLLVarAutoEncoder': NLLVarAutoEncoder,
            'Critic': Critic,
            'NLayerDiscriminator': NLayerDiscriminator,
            'VarAutoEncoder': VarAutoEncoder,
            'AutoEncoder': AutoEncoder,
            'VQVarAutoEncoder': VQVarAutoEncoder,
            'VQVarAutoEncoder': VQVarAutoEncoder,
            'Transformer': Transformer,
            'MultiScaleTransformer': MultiScaleTransformer,
            'TransformerCritic': TransformerCritic,
        }

    def __getitem__(self, item):
        try:
            return self.models[item]
        except KeyError:
            raise KeyError(f'No model named {item} found in registry')

    def __contains__(self, item):
        return item in self.models.keys()
