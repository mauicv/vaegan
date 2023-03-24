from duct.utils.config_mixin import ConfigMixin
from duct.utils.logging_mixin import LoggingMixin
from duct.model.transformer.mask import get_causal_mask
import torch
import shutil
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def train_step(model, opt, seq, mask):
    opt.zero_grad()
    seq_logits = model(seq, mask=mask)
    _, _, nt = seq_logits.shape
    loss = F.cross_entropy(
        seq_logits[:, :-1, :].reshape(-1, nt),
        seq[:, 1:].reshape(-1)
    )
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def assert_groups_vals(group1, group2):
    for item1, item2 in zip(group1, group2):
        for val1, val2 in zip(item1.values(), item2.values()):
            if isinstance(val1, list) and isinstance(val2, list):
                for a, b in zip(val1, val2):
                    assert a.shape == b.shape
            elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                assert val1.shape == val2.shape
            else:
                assert val1 == val2


def test_rel_emb_transformer(tmp_path):
    # tmp_path = './test_path'
    (tmp_path / 'test').mkdir()
    shutil.copyfile('./tests/test_configs/rel_emb_transformer.toml', str(tmp_path / 'test' / 'config.toml'))

    class Experiment(ConfigMixin, LoggingMixin):
        headers = ['transformer_opt']
        name = 'test'
        path = str(tmp_path)

    test_class = Experiment.init()
    test_class.setup_logs()
    _, mask = get_causal_mask(100)

    train_step(
        test_class.transformer, 
        test_class.transformer_opt, 
        torch.randint(0, 10, (64, 100)), 
        mask
    )

    test_class.save_state(tmp_path / 'model.pt')
    group1 = test_class.transformer.get_parameter_groups()
    test_class = Experiment.init()
    group2 = test_class.transformer.get_parameter_groups()
    assert_groups_vals(group1, group2)
    test_class.load_state(tmp_path / 'model.pt')

    train_step(
        test_class.transformer, 
        test_class.transformer_opt, 
        torch.randint(0, 10, (64, 100)), 
        mask
    )