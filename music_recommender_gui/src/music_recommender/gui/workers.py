"""Background threads for loading the dataset and training, so the window never freezes."""

import threading
import traceback

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from ..core import data
from ..core.training import Reporter, TrainingCancelled, run_training


class Worker(QObject):
    """Base class: subclasses implement work() and emit their own result signals.
    done is emitted last in every case so the thread can be stopped."""

    failed = pyqtSignal(str)
    done = pyqtSignal()

    def run(self):
        try:
            self.work()
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(str(exc) or details)
            traceback.print_exc()
        finally:
            self.done.emit()

    def work(self):
        raise NotImplementedError


class DatasetLoadWorker(Worker):
    loaded = pyqtSignal(object, str)    # DataFrame, path

    def __init__(self, path):
        super().__init__()
        self.path = str(path)

    def work(self):
        self.loaded.emit(data.load_dataset(self.path), self.path)


class _QtReporter(Reporter):
    def __init__(self, worker: "TrainingWorker"):
        self.worker = worker

    def stage(self, name, percent):
        self.worker.stageChanged.emit(name, percent)

    def progress(self, percent, detail=""):
        self.worker.progressChanged.emit(percent, detail)

    def log(self, text):
        self.worker.logLine.emit(text)

    def is_cancelled(self):
        return self.worker.cancel_event.is_set()


class TrainingWorker(Worker):
    stageChanged = pyqtSignal(str, int)
    progressChanged = pyqtSignal(int, str)
    logLine = pyqtSignal(str)
    trained = pyqtSignal(object)        # TrainedModel
    cancelled = pyqtSignal()

    def __init__(self, config, raw_df=None):
        super().__init__()
        self.config = config
        self.raw_df = raw_df
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def work(self):
        try:
            model = run_training(self.config, _QtReporter(self), raw_df=self.raw_df)
        except TrainingCancelled:
            self.cancelled.emit()
            return
        self.trained.emit(model)


def start_worker(worker: Worker, owner: QObject) -> QThread:
    """Run worker.run() in a new QThread that cleans itself up when the worker is done."""
    thread = QThread(owner)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.done.connect(thread.quit)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
