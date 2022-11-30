from duct.utils.logging_mixin import LoggingMixin
from duct.utils.save_imgs import save_img_pairs
import torch


def test_logger_csv(tmp_path):
    class Exp(LoggingMixin):
        path=str(tmp_path)
        name='test'
        headers=['test', 'test2']

    exp = Exp()
    exp.setup_logs()
    assert exp.log_dir.exists()
    assert exp.csv_filepath.exists()
    assert exp.imgs_path.exists()

    exp.log({'test': 1, 'test2': 2})

    with open(exp.csv_filepath, 'r') as f:
        assert f.readline() == 'test,test2\n'
        assert f.readline() == '1,2\n'

    assert exp.row_count == 1

    exp.log({'test': 1, 'test2': 2})
    for row in exp.load_logs():
        assert row['test'] == '1'
        assert row['test2'] == '2'

    assert exp.row_count == 2


def test_logger_imgs(tmp_path):
    class Exp(LoggingMixin):
        path=str(tmp_path)
        name='test'
        headers = ['test', 'test2']
        img_save_hook = save_img_pairs

    exp = Exp()
    exp.setup_logs()

    imgs_1  = torch.randn((6, 32, 32, 3))
    imgs_2  = torch.randn((6, 32, 32, 3))
    exp.save_imgs(imgs_1, imgs_2)

    assert exp.imgs_path.exists()
    assert len(list(exp.imgs_path.iterdir())) == 1
