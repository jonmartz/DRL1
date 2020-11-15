import os, sys


class Log:
    def __init__(self, _folder, _name):
        self.terminal = sys.stdout
        self.log_folder = _folder
        self.log_filename = _name
        if not os.path.exists(self.log_folder):
            print("NO LOG FOLDER FOUND - rebuiling it. ")
            os.makedirs(self.log_folder)
        self.full_filename = '{}/{}'.format(self.log_folder, self.log_filename)
        print('log file:{}'.format(self.full_filename))
        self.file_obj = open(self.full_filename, 'w+')

    def write(self, _str_to_write):
        self.terminal.write(_str_to_write)
        self.file_obj.write(_str_to_write)
        self.file_obj.flush()

    def close(self):
        self.file_obj.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.terminal.close()
        self.file_obj.close()
