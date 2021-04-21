import os
import time
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, save_dir):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        log_dir = save_dir + '/logs_{}'.format(time_str)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log = open(log_dir + '/log.txt', 'w')
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            self.log.write('{}: {}'.format(time_str, txt))
        else:
            self.log.write(txt)  
            self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()
    
    def close(self):
        self.log.close()
    
    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)


def main():
    logger = Logger('./')
    logger.write('test')
    logger.close()


if __name__ == '__main__':
    main()
