import os

class SuppressCStderr:
    def __enter__(self):
        self.stderr_fd = 2
        self.old_stderr = os.dup(self.stderr_fd)
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self.devnull, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.old_stderr, self.stderr_fd)
        os.close(self.devnull)
        os.close(self.old_stderr)