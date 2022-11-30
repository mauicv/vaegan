from pathlib import Path

class ExperimentBase:
    path = '.'

    def __init__(self):
        if not hasattr(self, 'name'):
            raise AttributeError('Experiment must have a name')

    @property
    def _path(self):
        return Path(self.path) / Path(self.name)
