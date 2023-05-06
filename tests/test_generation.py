from duct.utils.config_mixin import ConfigMixin
from duct.utils.logging_mixin import LoggingMixin
import torch
import shutil
from tests.util import set_seeds, disable_dropout
import pytest


def test_rel_emb_transformer_generation_1(tmp_path):
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


@pytest.mark.skip()
def test_rel_emb_transformer_generation_2(tmp_path):
    set_seeds()
    (tmp_path / 'test').mkdir()
    shutil.copyfile('./tests/test_configs/ret_gen.toml', str(tmp_path / 'test' / 'config.toml'))

    class Experiment(ConfigMixin, LoggingMixin):
        headers = ['transformer_opt']
        name = 'test'
        path = str(tmp_path)

    test_class = Experiment.init()
    seq = torch.randint(0, 256, (1, 512))
    disable_dropout(test_class.transformer)

    with torch.no_grad():
        seq_1_logits, _ = test_class.transformer.infer(seq)
        next_seq_1 = seq_1_logits.argmax(-1)[:, -1:]

        seq_2_logits = test_class.transformer(seq)
        next_seq_2 = seq_2_logits.argmax(-1)[:, -1:]

    print(next_seq_1, next_seq_2)

    assert torch.allclose(seq_1_logits, seq_2_logits)
    assert torch.all(next_seq_1 == next_seq_2)
