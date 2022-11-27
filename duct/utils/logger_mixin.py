from pathlib import Path
import csv


class LoggerMixin:
    headers = ()
    path='./logs'

    def __init__(self):
        self.path = Path(self.path)
        self.path.mkdir(exist_ok=True)
        self.img_path = self.path.joinpath('imgs')
        self.img_path.mkdir(exist_ok=True)
        self.csv_path = self.path.joinpath('data.csv')

        if not self.csv_path.exists():
            with open(self.csv_path, 'w') as log_f:
                writer = csv.DictWriter(log_f, fieldnames=self.headers)
                writer.writeheader()

    def log(self, logs):
        with open(self.logging_path_loss, 'a') as log_f:
            writer = csv.DictWriter(log_f, fieldnames=self.headers)
            writer.writerow(logs)