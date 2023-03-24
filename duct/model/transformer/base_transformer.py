import torch.nn as nn
import torch


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
        decay_params = {}
        no_decay_params = {}
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, module in self.named_modules():
            for pn, param in module.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if 'bias' in pn:
                    no_decay_params[fpn] = param
                elif 'weight' in pn and isinstance(module, whitelist_weight_modules):
                    decay_params[fpn] = param
                elif 'weight' in pn and isinstance(module, blacklist_weight_modules):
                    no_decay_params[fpn] = param

        # sort by name to ensure consistent ordering on reload
        decay_params = sorted(decay_params.items(), key=lambda x: x[0])
        decay_params = [p for _, p in decay_params]
        no_decay_params = sorted(no_decay_params.items(), key=lambda x: x[0])
        no_decay_params = [p for _, p in no_decay_params]
        return decay_params, no_decay_params

    def get_parameter_groups(self):
        decay_params, no_decay_params = self.get_weight_decay_params()
        groups = [
            {
                'params': decay_params,
                'weight_decay': 0.01,
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
            },
        ]
        return groups
