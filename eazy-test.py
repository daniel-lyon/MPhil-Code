from eazy import filters
from eazy.param import EazyParam
import os

res = filters.FilterFile('FILTER.RES.latest')

print(res.NFILT)
bp = res[400]

# for i in range(res.NFILT):
#     print(f'{i+1} {res.filters[i].name}')

params = EazyParam(PARAM_FILE='zphot.param.default')
print(params['Z_STEP'])