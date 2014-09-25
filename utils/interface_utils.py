import numpy as np
import sys

def prog_bar(t, N, length=20, bar_char='=', head_char='>'):
    sys.stdout.write('\r')
    format_str = '[%-'+str(length)+'s] %d%% (%d/%d)'
    sys.stdout.write(format_str %
                     (bar_char*int(float(t)/float(N)*length)+head_char,
                      np.ceil(float(t)/float(N)*100), t, N))
    sys.stdout.flush()
