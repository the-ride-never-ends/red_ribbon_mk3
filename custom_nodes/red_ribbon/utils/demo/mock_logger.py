# Example logger
def get_logger(name):
    class FakeLogger:
        def warning(self, msg): print(f"[WARNING] {msg}")
        def info(self, msg):    print(f"[INFO] {msg}")
    return FakeLogger()