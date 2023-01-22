import torch.nn as nn


class BaseTransformer():
    def __init__(self):
        super().__init__()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_weight_decay_params(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Linear):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        yield param

    def get_parameter_groups(self):
        decay_params = set(self.get_weight_decay_params())
        no_decay_params = [p for p in self.parameters() if p not in decay_params]
        groups = [
            {
                'params': list(decay_params),
                'weight_decay': 0.01,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
            },
        ]
        return groups
