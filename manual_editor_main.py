import os
import sys

from beadsketch.manual_editor import run, smoke_test


if __name__ == "__main__":
    if "--smoke-test" in sys.argv:
        smoke_test()
        os._exit(0)
    run()
