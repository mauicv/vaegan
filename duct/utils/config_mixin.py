import toml
from duct.model.model_registry import ModelRegistry
import torch
from duct.utils.experiment_base import ExperimentBase
from pathlib import Path
from datetime import datetime


def sort_by_date_strs(date_strs):
    return sorted(
        date_strs, 
        key=lambda x: datetime.strptime(str(x.stem), '%Y-%m-%d|%H:%M:%S')
    )


def load_config(path):
    return toml.load(path)


class ConfigMixin(ExperimentBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_registry = ModelRegistry()
        self.stateful_objs = []
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
        return self._path / 'models'

    def __getattr__(self, __name: str):
        if __name == 'cfg':
            return None
        elif getattr(self, 'cfg', None) is not None:
            return self.cfg.get(__name, None)
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
                    self.stateful_objs.append(key)
                    self.models.append(key)
                    if self.cfg.get('use_cuda', False):
                        model.cuda()

                    if opt_cfgs is not None:
                        for opt_cfg in opt_cfgs:
                            self.init_opt(model, opt_cfg)

    def init_opt(self, model, opt_cfg):
        opt_cls = opt_cfg.pop('class', None)
        opt_name = opt_cfg.pop('name', None)
        parameter_set_fn = opt_cfg.pop('parameter_set', None)
        opt_cls = getattr(torch.optim, opt_cls)
        if parameter_set_fn is not None:
            parameters = getattr(model, parameter_set_fn)()
        else:
            parameters = model.parameters()

        if opt_cls == 'AdamW':
            weight_decay = opt_cfg.pop('weight_decay', 0.01)
            parameters[0]['weight_decay'] = weight_decay

        setattr(self, opt_name, opt_cls(parameters, **opt_cfg))
        self.stateful_objs.append(opt_name)
        self.optimizers.append(opt_name)

    def load_state(self, path=None, replica='latest'):
        if path is None:
            path = self._model_path

        if getattr(self, 'num_saved_replicas'):
            path = path / f'{self._get_replica_path(path, replica)}'
        else:
            path = path / 'state.pt'

        data = torch.load(path)
        for key in self.stateful_objs:
            obj = getattr(self, key)
            obj.load_state_dict(data[key])
            if key in self.optimizers:
                for _, val in obj.state.items():
                    val['step'] = val['step'].cpu()

    def save_state(self, path=None):
        if path is None:
            path = self._model_path
            path.mkdir(parents=True, exist_ok=True)

        if getattr(self, 'num_saved_replicas'):
            date = datetime.now().strftime('%Y-%m-%d|%H:%M:%S')
            path = path / f'{date}.pt'
        else:
            path = path / 'state.pt'

        data = {
            key: getattr(self, key).state_dict() \
                for key in self.stateful_objs   
        }
        torch.save(data, path)

        if getattr(self, 'num_saved_replicas'):
            self._delete_old_replicas(path)

    def _delete_old_replicas(self, path):
        replicas = sort_by_date_strs(path.parent.glob('*'))
        if len(replicas) > self.num_saved_replicas:
            for replica in replicas[:-self.num_saved_replicas]:
                replica.unlink()

    def _get_replica_path(self, path, replica):
        replicas = sort_by_date_strs(path.glob('*'))
        if replica == 'latest':
            return replicas[-1]
        elif isinstance(replica, int):
            return replicas[-replica]