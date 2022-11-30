class ExperimentBase:
    def __init__(self):
        if not hasattr(self, 'name'):
            raise AttributeError('Experiment must have a name')
