from pathlib import Path
from duct.utils.experiment_base import ExperimentBase
import csv
import matplotlib.pyplot as plt


class LoggingMixin(ExperimentBase):
    headers = []
    img_save_hook = None

    def __init__(self):
        super().__init__()
        self.img_count = 0
        self.row_count = 0

    def setup_logs(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_filepath.touch(exist_ok=True)
        if self.csv_filepath.stat().st_size == 0:
            with self.csv_filepath.open('w') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                self.row_count = 0
        else:
            self.row_count = sum(1 for _ in self.load_logs())
        self.imgs_path.mkdir(parents=True, exist_ok=True)

    def log(self, dict):
        with self.csv_filepath.open('a') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(dict)
            self.row_count += 1

    def load_logs(self):
        with self.csv_filepath.open('r') as f:
            reader = csv.DictReader(f)
            self.row_count = 0
            for row in reader:
                self.row_count += 1
                yield row

    def save_imgs(self, *args, **kwargs):
        self.img_count += 1
        if self.img_save_hook is not None:
            self.img_save_hook(*args, **kwargs)

    @property
    def log_dir(self):
        return self._path / Path('logs')

    @property
    def csv_filepath(self):
        return self.log_dir / Path('experiment.csv')

    @property
    def imgs_path(self):
        return self.log_dir / Path('imgs')