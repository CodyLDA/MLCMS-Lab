import os.path
from datetime import datetime


class Logger:
    def __init__(self, log_filename=''):
        self.log = None
        self.counter_for_flush = 0

        if not log_filename:
            print('Error: Invalid address for log file')
            return

        open_mode = "a+"
        if not os.path.exists(log_filename):
            open_mode = "w+"
        self.log = open(os.path.dirname(__file__) + log_filename, open_mode)  # append mode

    def print_me(self, *args, date_and_time=True):
        print(args)
        if not self.log:
            print('warning: log file is not set!')
            return
        if date_and_time:
            now = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
            self.log.write('[' + now + '] ')
        for arg in args:
            self.log.write(str(arg) + ' ')
        self.log.write('\n')
        if self.counter_for_flush > 10:
            self.log.flush()
            self.counter_for_flush = 0
