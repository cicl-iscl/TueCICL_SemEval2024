import time


def timestart(name="Timer"):
    tic = time.perf_counter()
    def stop():
        toc = time.perf_counter()
        print(f"{name} took {toc - tic:0.4f} seconds")
    return stop