"""
lagrangian_parser.py
parser for lagrangian field data

"""
from __future__ import print_function

import os
import struct
import numpy as np


def parse_lagrangian_field(fn, returnAll=False, max_num_p=1e100, suppress=False):
    """
    parse internal field, extract data to numpy.array
    :param fn: file name
    :param returnAll: if variable is uniform, return array of len=Np instead of len=1
    :return: numpy array of internal field
    """
    if not os.path.exists(fn):
        if not suppress: print("Can not open file " + fn)
        return None
    with open(fn, "rb") as f:
        content = f.readlines()
        lag_field = parse_lagrangian_field_content(content, returnAll, max_num_p)
        del content
    return lag_field

def parse_lagrangian_field_content(content, returnAll, max_num_p):
    """
    parse internal field from content
    :param content: contents of lines
    :param returnAll: if variable is uniform, return array of len=Np instead of len=1
    :return: numpy array of internal field
    """
    is_binary = is_binary_format(content)
    for ln, lc in enumerate(content):
        line = lc.decode('utf-8').split()
        if len(line) > 0:
            uniformStr = line[0].split("{")
            if uniformStr[0].isdigit() and b"{" in lc and b"}" in lc:
                return parse_data_uniform(content[ln], returnAll, max_num_p)
                break
            if line[0].isdigit():
                return parse_data_nonuniform(content, ln, len(content), is_binary, max_num_p)
                # elif b'uniform' in lc:
                #     return parse_data_uniform(content[ln])
                break
    return None

def parse_data_uniform(line, returnAll, max_num_p):
    """
    parse uniform data from a line
    :param line: a line include uniform data, eg. "635625{-1}"
    :return: data
    """

    if b'{(' in line:
        data = np.array([float(x) for x in line.split(b'{(')[1].split(b')}')[0].split()])
        #data = np.array([float(x) for x in line.split(b'{')[1].split(b'}')[0].split()])
        if returnAll:
            numP = min(int( line.split(b'{')[0] ), int(max_num_p))
            if data.shape[0]==3:
                return data*np.ones((numP,3))
            elif data.shape[0]==1:
                return data*np.ones((numP, 1))
        else:
            return data
    elif not b'{(' in line and b'{' in line:
        #data = np.array([float(x) for x in line.split(b'{(')[1].split(b')}')[0].split()])
        data = np.array([float(x) for x in line.split(b'{')[1].split(b'}')[0].split()])
        if returnAll:
            numP = min(int( line.split(b'{')[0] ), int(max_num_p))
            return data*np.ones((numP, 1))
        else:
            return data
    return "ERROR in parse_data_uniform"#float(line.split(b'uniform')[1].split(b';')[0])


def parse_data_nonuniform(content, n, n2, is_binary, max_num_p):
    """
    parse nonuniform data from lines
    :param content: data content
    :param n: line number
    :param n2: last line number
    :param is_binary: binary format or not
    :return: data
    """
    num = min(int(content[n]), int(max_num_p))
    if not is_binary:
        if b'scalar' in content[n]:
            data = np.array([float(x) for x in content[n + 2:n + 2 + num]])
        else:
            i = [ln[1:-2].split() for ln in content[n + 2:n + 2 + num]]
            tmpStr = content[n+3].replace(b'(', b'').replace(b')', b'')
            # data = np.array([ln[1:-2].split() 
            #                     for ln in content[n + 2:n + 2 + num]], dtype=float)
            data = np.array([ln[:-1].replace(b'(', b'').replace(b')', b'').split()
                                for ln in content[n + 2:n + 2 + num]], dtype=float)
    else:
        nn = 1
        if b'vector' in content[n]:
            nn = 3
        elif b'symmtensor' in content[n]:
            nn = 6
        elif b'tensor' in content[n]:
            nn = 9
        buf = b''.join(content[n+2:n2+1])
        vv = np.array(struct.unpack('{}d'.format(num*nn),
                                    buf[struct.calcsize('c'):num*nn*struct.calcsize('d')+struct.calcsize('c')]))
        if nn > 1:
            data = vv.reshape((num, nn))
        else:
            data = vv
    return data

def is_binary_format(content, maxline=20):
    """
    parse file header to judge the format is binary or not
    :param content: file content in line list
    :param maxline: maximum lines to parse
    :return: binary format or not
    """
    for lc in content[:maxline]:
        if b'format' in lc:
            if b'binary' in lc:
                return True
            return False
    return False

