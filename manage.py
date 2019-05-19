import inspect
import sys
import traceback
from datetime import datetime
from enum import Enum
from functools import wraps

from src.bayesian_network.bayesian_network_manager import BayesianNetworkManager
from src.config.runtime_config import RuntimeConfig
from src.lda.lda_manager import LDAManager
from src.pca.pca_manager import PCAManager

__author__ = 'e.sadeqi'


class ProcessStatusType(Enum):
    started = 1
    finished = 2


def process_status_printer(status: ProcessStatusType, function_name=None):
    time_format = '%Y-%m-%d %H:%M:%S'
    current_frame = inspect.currentframe()
    outer_frame = inspect.getouterframes(current_frame, 2)
    if function_name is None:
        function_name = outer_frame[1][3]
    current_time = datetime.now().strftime(time_format)
    print(current_time, function_name, status.name, sep='\t')


def process_wrapper(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            process_status_printer(ProcessStatusType.started, f.__name__)
            f(*args, **kwargs)
            process_status_printer(ProcessStatusType.finished, f.__name__)

        except Exception as e:
            if RuntimeConfig.DEBUG_MODE is True:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type, exc_value, exc_traceback,
                                          limit=2, file=sys.stdout)
            else:
                raise e

    return wrap


@process_wrapper
def pca():
    manager = PCAManager()
    manager.solve()
    print()


@process_wrapper
def lda():
    manager = LDAManager()
    manager.solve()
    print()


@process_wrapper
def bayes(*args):
    method = None
    if args:
        method, *_ = args
    manager = BayesianNetworkManager()
    manager.run_with_cross_validation(method=method)


if __name__ == '__main__':
    argv = sys.argv
    available_function_list = [
        pca,
        lda,
        bayes,
    ]
    function_names = ""
    for function_ in available_function_list:
        function_names += "\t{},\n".format(function_.__name__)
    man_msg = (
        "Possible commands are [\n{command_names}]\n\n"
        "Special thanks to {author} :D"
    ).format(
        command_names=function_names,
        author=__author__
    )

    if len(argv) < 2:
        print("Empty command")
        print(man_msg)
        exit(-1)

    _, command, *arguments = argv
    command = str(command).strip()

    for function_ in available_function_list:
        if command == function_.__name__:
            function_(*arguments)
            exit(0)
    else:
        print("Wrong command:\t" + command)
        print(man_msg)
