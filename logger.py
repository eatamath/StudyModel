import pickle
import time
import os


def current_time():
    tm = time.ctime()
    return tm.replace(' ', '-').replace(':', '-')


def write_logs(writer, logs, stage='train'):
    for k, v in logs.items():
        writer.add_scalar(stage + '-' + k, v)
    return


class MyLogger:
    def __init__(self, path='./data/', stage='train', auto_flush=True):
        self.stage_logs = []
        self.path = path
        self.stage = stage
        self.tm = current_time()
        self.auto_flush = auto_flush
        return

    def append(self, log):
        self.stage_logs.append(log)
        if self.auto_flush:
            self.flush()
        return

    def flush(self):
        with open(os.path.join(self.path, 'log-' + self.stage + '-' + self.tm + '.txt'), 'wb') as f:
            pickle.dump(self.stage_logs, f)
        return
