import toml
from duct.model.model_registry import ModelRegistry
import torch
from duct.utils.experiment_base import ExperimentBase
from pathlib import Path


def load_config(path):
    return toml.load(path)


class ConfigMixin(ExperimentBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_registry = ModelRegistry()
        self.sateful_objs = []
        self.models = []
        self.optimizers = []
        self.make()

    @classmethod
    def init(cls):
        assert cls.path is not None, \
            'path must be set on class'
        path = Path(cls.path) / Path(cls.name) / 'config.toml'
        return cls(load_config(path))

    @classmethod
    def from_file(cls, path=None):
        return cls(load_config(path))

    @classmethod
    def from_toml(cls, toml_data_str):
        return cls(toml.loads(toml_data_str))

    @property
    def _model_path(self):
        return self._path / 'models' / 'state.pt'

    def __getattr__(self, __name: str):
        if __name == 'cfg':
            return None
        elif getattr(self, 'cfg', None) is not None:
            return self.cfg[__name]
        raise AttributeError("can't get attribute")

    def check_if_model_cfg(self, cfg):
        if isinstance(cfg, dict) and 'class' in cfg:
            if cfg['class'] in self.model_registry:
                return True
        return False

    def make(self):
        for key in self.cfg:
            if self.check_if_model_cfg(self.cfg[key]):
                model_cfg = self.cfg[key]
                cls = model_cfg.get('class', None)
                if cls in self.model_registry:
                    cls = model_cfg.pop('class')
                    opt_cfgs = model_cfg.pop('opt_cfgs', None)
                    model = self.model_registry[cls]
                    model = model(**model_cfg)
                    setattr(self, key, model)
                    self.sateful_objs.append(key)
                    self.models.append(key)
                    if self.cfg.get('use_cuda', False):
                        model.cuda()

                    if opt_cfgs is not None:
                        for opt_cfg in opt_cfgs:
                            self.init_opt(model, opt_cfg)

    def init_opt(self, model, opt_cfg):
        opt_cls = opt_cfg.pop('class', None)
        opt_name = opt_cfg.pop('name', None)
        paramter_set_fn = opt_cfg.pop('parameter_set', None)
        opt_cls = getattr(torch.optim, opt_cls)
        if paramter_set_fn is not None:
            parameters = getattr(model, paramter_set_fn)()
        else:
            parameters = model.parameters()
        setattr(self, opt_name, opt_cls(parameters, **opt_cfg))
        self.sateful_objs.append(opt_name)
        self.optimizers.append(opt_name)

    def load_state(self, path=None):
        if path is None:
            path = self._model_path
        data = torch.load(path)
        for key in self.sateful_objs:
            obj = getattr(self, key)
            obj.load_state_dict(data[key])

            if key in self.optimizers:
                for _, val in obj.state.items():
                    val['step'] = val['step'].cpu()

    def save_state(self, path=None):
        if path is None:
            path = self._model_path
            path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            key: getattr(self, key).state_dict() \
                for key in self.sateful_objs   
        }
        torch.save(data, path)

