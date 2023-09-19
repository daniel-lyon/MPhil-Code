import numpy as np
import matplotlib.pyplot as plt

from eazy.templates import Redden

fig, ax = plt.subplots(1,1,figsize=(6,4))

wave = np.arange(1200, 2.e4)

for model in ['calzetti00', 'mw', 'smc', 'reddy15']:
    redfunc = Redden(model=model, Av=1.0)
    ax.plot(wave, redfunc(wave), label=model)

ax.plot(wave, wave*0+10**(-0.4), color='k', 
          label=r'$A_\lambda = 1$', linestyle=':')

ax.legend()
ax.loglog()

ax.set_xticks([2000, 5000, 1.e4])
ax.set_xticklabels([0.2, 0.5, 1.0])

ax.grid()
ax.set_xlabel('wavelength, microns')
ax.set_ylabel('Attenuation / extinction (Av=1 mag)')

fig.tight_layout(pad=0.5)
plt.show()