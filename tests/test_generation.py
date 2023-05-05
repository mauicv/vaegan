from duct.utils.config_mixin import ConfigMixin
from duct.utils.logging_mixin import LoggingMixin
import torch
import shutil


def test_rel_emb_transformer(tmp_path):
    (tmp_path / 'test').mkdir()
    shutil.copyfile('./tests/test_configs/ret_gen.toml', str(tmp_path / 'test' / 'config.toml'))

    class Experiment(ConfigMixin, LoggingMixin):
        headers = ['transformer_opt']
        name = 'test'
        path = str(tmp_path)

    test_class = Experiment.init()
    seq = torch.randint(0, 256, (1, 509))
    ps = None
    for i in range(5):
        seq, ps = test_class.transformer.infer(seq, ps)
        seq = seq.argmax(-1)[:, -1:]
        assert seq.shape == (1, 1)
        assert ps[0]['prev_k'].shape == \
            (1, 12, min(509 + i, 511), 64)