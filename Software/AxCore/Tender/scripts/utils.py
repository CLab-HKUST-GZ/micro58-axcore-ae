import os

def opt_decomp_params(size, bits):
    if size == "6.7b":
        if bits == 4:
            decomp = 8
        elif bits == 8:
            decomp = 4
    elif size == "13b":
        if bits == 4:
            decomp = 8
        elif bits == 8:
            decomp = 4
    elif size == "30b":
        if bits == 4:
            decomp = 8
        elif bits == 8:
            decomp = 4
    elif size == "66b":
        if bits == 4:
            decomp = 10
        elif bits == 8:
            decomp = 8
    else:
        raise ValueError
    return decomp

def llama2_decomp_params(size, bits):
    if size == "7b":
        if bits == 4:
            decomp = 14
        elif bits == 8:
            decomp = 8
    elif size == "13b":
        if bits == 4:
            decomp = 16
        elif bits == 8:
            decomp = 14
    elif size == "70b":
        if bits == 4:
            decomp = 20
        elif bits == 8:
            decomp = 16
    else:
        raise ValueError
    return decomp


def set_symlink_opt(name):
    if not os.path.exists(f'../models/{name}'):
        print(f'no such file in ../models/{name}')
        exit(1)

    if os.path.exists(f'../models/modeling_opt.py'):
        os.system(f'rm ../models/modeling_opt.py')

    os.system(f'ln -s ../models/{name} ../models/modeling_opt.py')


def set_symlink_llama(name):
    if not os.path.exists(f'../models/{name}'):
        print(f'no such file in ../models/{name}')
        exit(1)

    if os.path.exists(f'../models/modeling_llama.py'):
        os.system(f'rm ../models/modeling_llama.py')

    os.system(f'ln -s ../models/{name} ../models/modeling_llama.py')
