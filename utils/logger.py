import tensorflow as tf
import os

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.log_dir = log_dir
        self.log_file_path = os.path.join(log_dir, 'log.txt')
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        with self.writer.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        with open(self.log_file_path, mode='a') as f:
            f.write(f'step{step}\n')
            for tag, value in tag_value_pairs:
                f.write(f'{tag}: {value}\n')
