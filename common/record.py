import os
import sys


def record_setting(out):
    out = out.split()[0]
    if not os.path.exists(out):
        os.mkdir(out)

    with open(out + '/command.txt', 'w') as f:
        f.write(" ".join(sys.argv) + "\n")
