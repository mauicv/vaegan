from pathlib import Path
from duct.utils.experiment_base import ExperimentBase
import csv
import matplotlib.pyplot as plt
from datetime import datetime


class LoggingMixin(ExperimentBase):
    headers = []
    save_hook = None

    def __init__(self):
        super().__init__()
        self.iter_count = 0
        self.row_count = 0

    def setup_logs(self):
        self.headers = ['datetime', *self.headers]
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_filepath.touch(exist_ok=True)
        if self.csv_filepath.stat().st_size == 0:
            with self.csv_filepath.open('w') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                self.row_count = 0
        else:
            self.row_count = sum(1 for _ in self.load_logs())
        self.training_artifcat_path.mkdir(parents=True, exist_ok=True)

    def log(self, dict):
        dict['datetime'] = datetime.now().strftime('%Y-%m-%d|%H:%M:%S')
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

    def save_training_artifacts(self, *args, **kwargs):
        self.iter_count += 1
        if self.save_hook is not None:
            self.save_hook(*args, **kwargs)

    @property
    def log_dir(self):
        return self._path / Path('logs')

    @property
    def csv_filepath(self):
        return self.log_dir / Path('experiment.csv')

    @property
    def training_artifcat_path(self):
        return self.log_dir / Path('training_artifacts')