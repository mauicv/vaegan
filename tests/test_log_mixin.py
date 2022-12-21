from duct.utils.logging_mixin import LoggingMixin
from duct.utils.save_imgs import save_img_pairs
from duct.utils.audio import save_audio
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
    assert exp.training_artifcat_path.exists()

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
        save_hook = save_img_pairs

    exp = Exp()
    exp.setup_logs()

    imgs_1  = torch.randn((6, 32, 32, 3))
    imgs_2  = torch.randn((6, 32, 32, 3))
    exp.save_training_artifacts(imgs_1, imgs_2)

    assert exp.training_artifcat_path.exists()
    assert len(list(exp.training_artifcat_path.iterdir())) == 1


def test_logger_audio(tmp_path):
    class Exp(LoggingMixin):
        path=str(tmp_path)
        name='test'
        headers = ['test', 'test2']
        save_hook = save_audio

    exp = Exp()
    exp.setup_logs()

    aud_1  = torch.randn((2, 128))
    aud_2  = torch.randn((2, 128))
    exp.save_training_artifacts(aud_1, aud_2, 1)

    assert exp.training_artifcat_path.exists()
    assert len(list(exp.training_artifcat_path.iterdir())) == 1
