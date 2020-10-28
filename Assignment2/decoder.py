'''
Author: Manas Vashistha
'''

import numpy as np
import sys

def decode(l):
    if l == 0:
        return 'N'
    elif l == 1:
        return 'E'
    elif l == 2:
        return 'S'
    elif l == 3:
        return 'W'

if __name__ == '__main__':
    opts = sys.argv[1::2]
    args = sys.argv[2::2]

    for  i in range(len(opts)):
        opt = opts[i]
        arg = args[i]
        if opt == '--grid':
            gridfile = str(arg)
        if opt == '--value_policy':
            value_and_policy_file = str(arg)

    path = []
    start = -1
    end = []
    state = {}
    i = -1
    j = -1
    k = -1

    f = open(gridfile, 'r')
    for line in f:
        i += 1
        j = -1
        bits = line.strip().split(' ')
        for b in bits:
            j += 1
            temp = int(b)
            if temp == 1:
                continue
            k += 1
            state[(i, j)] = k
            if temp == 2:
                start = state[(i, j)]
            if temp == 3:
                end.append(state[(i, j)])      
    f.close()


    f = open(value_and_policy_file, 'r')
    for line in f:
        if line != '\n':
            l = int(line.strip().split('\t')[1])
            path.append(decode(l))
    f.close()

    spath = [path[start]]
    sstates = [start]
    keys = list(state.keys())
    vals = list(state.values())

    while True:
        if sstates[-1] in end:
            spath = spath[:-1]
            break
        if spath[-1] == 'N':
            spath.append(path[state[(keys[vals.index(sstates[-1])][0] - 1, keys[vals.index(sstates[-1])][1])]])
            sstates.append(state[(keys[vals.index(sstates[-1])][0] - 1, keys[vals.index(sstates[-1])][1])])
        if spath[-1] == 'E':
            spath.append(path[state[(keys[vals.index(sstates[-1])][0], keys[vals.index(sstates[-1])][1] + 1)]])
            sstates.append(state[(keys[vals.index(sstates[-1])][0], keys[vals.index(sstates[-1])][1] + 1)])
        if spath[-1] == 'S':
            spath.append(path[state[(keys[vals.index(sstates[-1])][0] + 1, keys[vals.index(sstates[-1])][1])]])
            sstates.append(state[(keys[vals.index(sstates[-1])][0] + 1, keys[vals.index(sstates[-1])][1])])
        if spath[-1] == 'W':
            spath.append(path[state[(keys[vals.index(sstates[-1])][0], keys[vals.index(sstates[-1])][1] - 1)]])
            sstates.append(state[(keys[vals.index(sstates[-1])][0], keys[vals.index(sstates[-1])][1] - 1)])

    for p in spath:
        sys.stdout.write(p + ' ')
        sys.stdout.flush()
        
