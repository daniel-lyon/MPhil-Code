import numpy as np

unit_prefixes = {
    -30 : 'q',
    -27 : 'r',
    -24 : 'y', 
    -21 : 'z',
    -18 : 'a',
    -15 : 'f',
    -12 : 'p',
    -9 : 'n',
    -6 : '\u03BC',
    -3 : 'm',
    0 : '', 
    3 : 'k',
    6 : 'M',
    9 : 'G',
    12 : 'T',
    15 : 'P',
    18 : 'E', 
    21 : 'Z',
    24 : 'Y',
    27 : 'R',
    30 : 'Q',
}

def get_sci_exponent(number):
    """ Find the scientific exponent of a number """
    abs_num = np.abs(number)
    base = np.log10(abs_num)  # Log rules to find exponent
    exponent = int(np.floor(base))  # convert to floor integer
    return exponent

def get_eng_exponent(number):
    """ 
    Find the nearest power of 3 (lower). In engineering format,
    exponents are multiples of 3.
    """
    exponent = get_sci_exponent(number)  # Get scientific exponent
    for i in range(3):
        if exponent > 0:
            unit = exponent-i
        else:
            unit = exponent+i
        if unit % 3 == 0:  # If multiple of 3, return it
            return unit