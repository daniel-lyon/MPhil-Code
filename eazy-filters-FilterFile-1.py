import matplotlib.pyplot as plt
from eazy.filters import FilterFile

res = FilterFile('FILTER.RES.latest')
print(len(res.filters))

bp = res[205]
print(bp)

fig, ax = plt.subplots(1,1,figsize=(6,4))

ax.plot(bp.wave, bp.throughput, label=bp.name.split()[0])

ax.set_xlabel('wavelength, Angstroms')
ax.set_ylabel('throughput')
ax.legend()
ax.grid()

fig.tight_layout(pad=0.5)
plt.show()