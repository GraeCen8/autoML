import time

def timeCounter(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        total = (end - start).__floor__()
        return result, total
    return wrap
