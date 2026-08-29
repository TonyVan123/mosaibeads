import os
import sys

from beadsketch.excel_converter import run, smoke_test


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
        os._exit(0)
    initial = next((arg for arg in sys.argv[1:] if arg.lower().endswith(".xlsx")), None)
    run(initial)
