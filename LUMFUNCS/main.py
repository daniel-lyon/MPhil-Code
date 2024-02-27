import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM

class LF(object):
    def __init__(self, cosmo, z, M_abs, z_bins, lum_bins, mlim, survey_area, min_count=0, nrows=3, ncols=2, ylim=(-10, -2), xlabel='$M_{AB}$'):
        """ 
        Class to calculate the luminosity/magnitude function of a galaxy sample.
        
        Parameters
        ----------
        cosmo : astropy.cosmology.FlatLambdaCDM
            Cosmology object.
        
        z : array-like
            Redshift of the galaxies.
        
        M_abs : array-like
            Absolute magnitude of the galaxies.
        
        z_bins : list of tuples
            Redshift bins. Example: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        
        lum_bins : array-like
            Luminosity bins. Example: np.arange(-25, -10, 0.5)
        
        mlim : float
            Apparent magnitude limit of the survey.
        
        survey_area : float
            Survey area in square degrees.
            
        min_count : int, optional
            Minimum number of galaxies in a luminosity bin to be plotted. Default is 0.
        
        nrows : int, optional
            Number of rows in the plot. Default is 3.
        
        ncols : int, optional
            Number of columns in the plot. Default is 2.
            
        ylim : tuple, optional
            Y-axis limits of the plot. Default is (-10, -2).
        
        xlabel : str, optional
            X-axis label of the plot. Default is '$M_{AB}$'. Options are '$M_{AB}$' or 'log($L_{IR}$ [$L_{\odot}$])'.
        """
        self._cosmo = cosmo
        self._z = z
        self._z_bins = z_bins
        self._lum_bins = lum_bins
        self._mlim = mlim
        self._survey_area = survey_area
        
        self._min_count = min_count
        self._nrows = nrows
        self._ncols = ncols
        self._ylim = ylim
        self._xlabel = xlabel
          
        df = pd.DataFrame({
            'z': self._z,
            'lum': M_abs
        })
        self._df = df        
    
    def _bin_data(self):
        """ 
        Bin the data by redshift and luminosity.
        """
        all_bins = []
        
        # Bin the data by redshift
        for min_z, max_z in self._z_bins:
            z_mask = (self._z >= min_z) & (self._z < max_z)
            binned_z_data = self._df[z_mask]       
                
            # Bin the data by luminosity
            counts, lum_bin_edges = np.histogram(binned_z_data['lum'], bins=self._lum_bins)
            lum_bin_centers = lum_bin_edges[0:-1] + (np.diff(lum_bin_edges)[0] / 2)
            
            # Calculate the volume of each luminosity bin
            volumes = self._calc_volume(lum_bin_centers, min_z, max_z)
            
            # Save the counts, volumes, and luminosity bin centers
            all_bins.append([counts, volumes, lum_bin_centers, binned_z_data['lum']])
        return all_bins
    
    def _calc_volume(self, lum_bin_centers, zbin_min, zbin_max):
        """ 
        Calculate the volume of each luminosity bin.
        """
        # Calculate the minimum and maximum distance of the redshift bin
        dmin = self._cosmo.comoving_distance(zbin_min).value # Mpc
        dmax = self._cosmo.comoving_distance(zbin_max).value # Mpc

        # Calculate the maximum distance of each luminosity bin
        if type(self._mlim) == (int or float):
            dmaxs = (10 * 10 ** ((self._mlim - lum_bin_centers) / 5)) # pc
            dmaxs = dmaxs / 10 ** 6 # pc -> Mpc
        elif type(self._mlim) == list:
            max_z = 0.652 * (10 ** ((lum_bin_centers - 6.586) / 5.336) - 0.768)
            dmaxs = self._cosmo.comoving_distance(max_z).value # Mpc

        # If the maximum distance is greater than the maximum redshift bin distance, set it to the maximum redshift bin distance
        dmaxs[dmaxs > dmax] = dmax

        # Calculate the minimum volume of the redshift bin
        vmin = 4/3 * np.pi * dmin**3 # Mpc^3

        # Calculate the maximum volume of each luminosity bin
        vmaxs = 4/3 * np.pi * dmaxs**3 # Mpc^3

        # Total volume probed accounting for survey area
        vol = (vmaxs - vmin) * (self._survey_area / 41253) # Mpc^3
        return vol
    
    def plot(self):
        """ 
        Make a simple plot of the luminosity function with no fitting.
        """
        data = self._bin_data()
        _, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        
        # Plot the luminosity function for each redshift bin
        for ax, d, (z_start, z_end) in zip(axes.flatten(), data, self._z_bins):
            counts, volumes, lum_bin_centers, _ = d
            
            # Mask the data if the number of galaxies in a luminosity bin is less than min_count
            mask = counts > self._min_count
            lf = counts[mask] / volumes[mask]
            lum_bin_centers = lum_bin_centers[mask]
            
            # Plot the luminosity function
            ax.scatter(lum_bin_centers, np.log10(lf), label=f'{z_start} <= z < {z_end}')
            ax.set_xlabel('$M_{AB}$')
            ax.set_ylabel('log(phi)')
            ax.legend()
            
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _schechter_magnitude(M, M_star, phi_star, alpha):
        return phi_star * 10 ** (-0.4*(1-alpha)*(M-M_star)) * np.exp(-10.**(-0.4*(M-M_star)))
    
    @staticmethod
    def _schecter_luminosity(L, L_star, phi_star, alpha):
        return phi_star * 10 ** (-0.4*(1-alpha)*(L_star-L)) * np.exp(-10.**(-0.4*(L_star-L))) 
    
    @staticmethod
    def _saunders_magnitude(M, M_star, phi_star, alpha, sigma):
        return phi_star * 10 ** (-0.4*(1-alpha)*(M-M_star)) * np.exp(-1 / (2 * sigma ** 2) * (np.log10(1 + 10 ** (-0.4 * (M - M_star))))**2)
    
    @staticmethod
    def _saunders_luminosity(L, L_star, phi_star, alpha, sigma):
        return phi_star * 10 ** (-0.4*(1-alpha)*(L_star-L)) * np.exp(-1 / (2 * sigma ** 2) * (np.log10(1 + 10 ** (-0.4 * (L_star - L))))**2)
       
    def _get_params(self, func, lum_bin_centers, lf, z_start, z_end, maxfev=1000, p0=None):
        """ 
        Get the parameters of the function fit.
        """
        if p0 is None and func == 'Schechter':
            p0 = [lum_bin_centers[0], 0.001, -0.9]
            f = self._schechter_magnitude
        elif p0 is None and func == 'Schechter_lum':
            p0 = [lum_bin_centers[-1], 0.001, -0.9]
            f = self._schecter_luminosity
        elif p0 is None and func == 'Saunders':
            p0 = [lum_bin_centers[0], 0.001, -0.9, 0.1]
            f = self._saunders_magnitude
        elif p0 is None and func == 'Saunders_lum':
            p0 = [lum_bin_centers[-1], 0.001, -0.9, 0.1]
            f = self._saunders_luminosity
        
        # Fit the function to the data
        try:
            params, _ = curve_fit(f, lum_bin_centers, lf, p0=p0, maxfev=maxfev)
        except RuntimeError:
            warnings.warn(f'Optimal parameters not found {z_start} <= z < {z_end}: Number of calls to function has reached maxfev = {maxfev}', RuntimeWarning)
            return False
        except TypeError:
            warnings.warn(f'TypeError: The number of func parameters={len(p0)} must not exceed the number of data points={len(lum_bin_centers)}', RuntimeWarning)
            return False
        except ValueError:
            warnings.warn(f'ValueError: mask has no data between {z_start} <= z < {z_end}', RuntimeWarning)
            return False
        return params
        
    def fit(self, func, maxfev=1000, p0=None, verbose=True):
        """ 
        Fit a given function to the luminosity function.
        
        Parameters
        ----------
        func : str
            The function to fit. Options are 'Schechter', 'Schechter_lum', 'Saunders', or 'Saunders_lum'.
        
        maxfev : int, optional
            Maximum number of calls to function. Default is 1000.
        
        p0 : array-like, optional
            Initial guess for the parameters. Default 'None' will auto set.
            
        verbose : bool, optional
            Print the fit parameters. Default is True.
        """
        data = self._bin_data()
        fig, axes = plt.subplots(self._nrows, self._ncols, figsize=(15, 10), sharex=True, sharey=True)

        # If there is more than one plot, flatten the axes
        try:
            axes = axes.flatten()
        except:
            axes = [axes]
        
        # Choose the function to fit
        if func == 'Schechter':
            f = self._schechter_magnitude
        elif func == 'Saunders':
            f = self._saunders_magnitude
        elif func == 'Schechter_lum':
            f = self._schecter_luminosity
        elif func == 'Saunders_lum':
            f = self._saunders_luminosity
        
        # For each redshift bin, fit the function to the data
        for ax, d, (z_start, z_end) in zip(axes, data, self._z_bins):
            counts, volumes, lum_bin_centers, _ = d
            
            # Mask the data if the number of galaxies in a luminosity bin is less than min_count
            mask = counts > self._min_count
            lf = counts[mask] / volumes[mask]
            lum_bin_centers = lum_bin_centers[mask]
            
            # Skip the redshift bin if all the data is masked
            if len(lum_bin_centers) == 0:
                continue
            
            # Calculate the error in the luminosity function
            upp_lf, low_lf = self._phi_error(counts[mask], volumes[mask])
            
            # Fit the function to the data
            params = self._get_params(func, lum_bin_centers, lf, z_start, z_end, maxfev, p0)
            if params is False:
                continue
                        
            # Print the fit parameters
            if verbose:
                print(f'{z_start} <= z < {z_end} Function fit:')
                print(f'M_star = {params[0]:.2e}')
                print(f'phi_star = {params[1]:.2e}')
                print(f'alpha = {params[2]:.2e}')
                if len(params) == 4:
                    print(f'sigma = {params[3]:.2e}')
                print('\n')
                
            upper_params = self._get_params(func, lum_bin_centers, upp_lf, z_start, z_end, maxfev, p0)
            lower_params = self._get_params(func, lum_bin_centers, low_lf, z_start, z_end, maxfev, p0)
            
            # Plot the luminosity function and the fit
            ax.errorbar(lum_bin_centers, np.log10(lf), yerr=[np.log10(lf) - np.log10(low_lf), np.log10(upp_lf) - np.log10(lf)], 
                fmt='o', label=f'{z_start} $\leq$ z < {z_end}', capsize=4)
            
            ax.plot(lum_bin_centers, np.log10(f(lum_bin_centers, *params)), label=f'{func} fit', color='red', linestyle='--')
            
            long_lum_smooth = np.linspace(lum_bin_centers[0], lum_bin_centers[-1], 100)
            if type(lower_params) != bool and type(upper_params) != bool:
                ax.fill_between(long_lum_smooth, 
                    np.log10(f(long_lum_smooth, *lower_params)), 
                    np.log10(f(long_lum_smooth, *upper_params)), 
                    alpha=0.2, color='red', label='Fit Error')
                
            # ax.fill_between(lum_bin_centers, np.log10(low_lf), np.log10(upp_lf), alpha=0.2, color='magenta', label='Error')
            ax.set_ylim(self._ylim)
            ax.legend()
            # ax.legend(loc='lower left')
        
        # fig.suptitle('ZFOURGE Daniel Code')
        fig.supylabel('log($\phi$ $Mpc^-3$)')
        fig.supxlabel(self._xlabel)
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.show()
    
    @staticmethod
    def _phi_error(counts, volumes):
        """ 
        Calculate the error in the luminosity function using Poisson statistics.
        """
        upper_counts = counts + np.sqrt(counts)
        lower_counts = counts - np.sqrt(counts)
        
        upper_phi = upper_counts / volumes
        lower_phi = lower_counts / volumes
        return upper_phi, lower_phi
        
    def print_counts(self):
        """ 
        Print the number of galaxies in each luminosity bin.
        """
        data = self._bin_data()
        print('Number of galaxies in each luminosity bin:')
        for d, (z_start, z_end) in zip(data, self._z_bins):
            counts, _, _, _ = d
            print(f'{z_start} <= z < {z_end}: {counts.tolist()}. Total = {np.sum(counts)}')
        print('\n')
    
    def print_volumes(self):
        """ 
        Print the volume of each luminosity bin.
        """
        data = self._bin_data()
        print('Volume of each luminosity bin:')
        for d, (z_start, z_end) in zip(data, self._z_bins):
            _, volumes, _, _ = d
            volumes = [f'{v:.2e}' for v in volumes]
            print(f'{z_start} <= z < {z_end}: {volumes}')
        print('\n')
    
    def overlay_plot(self):
        """ 
        Make a simple plot of the luminosity function with no fitting.
        """
        data = self._bin_data()
        
        # Plot the luminosity function for each redshift bin
        for d, (z_start, z_end) in zip(data, self._z_bins):
            counts, volumes, lum_bin_centers, _ = d
            
            # Mask the data if the number of galaxies in a luminosity bin is less than min_count
            mask = counts > self._min_count
            lf = counts[mask] / volumes[mask]
            lum_bin_centers = lum_bin_centers[mask]
            
            # Plot the luminosity function
            plt.plot(lum_bin_centers, np.log10(lf), label=f'{z_start} $\leq$ z < {z_end}')
        
        plt.legend()
        plt.tight_layout()
        plt.xlabel(self._xlabel)
        plt.ylabel('log(phi)')
        plt.show()
        
    def plot_histograms(self):
        data = self._bin_data()
        
        fig, axes = plt.subplots(self._nrows, self._ncols, figsize=(15, 10), sharex=True)
        
        for ax, d, (z_start, z_end) in zip(axes.flatten(), data, self._z_bins):
            _, _, _, lum = d
            ax.hist(lum, bins=self._lum_bins, histtype='step', label=f'{z_start} $\leq$ z < {z_end}')
            ax.axhline(y=self._min_count, color='red', linestyle='--', label='Mask Threshold')
            # ax.fill_between(self._lum_bins, 0, self._min_count, alpha=0.2, color='red')
            ax.legend()
        
        fig.supylabel('Number of Galaxies')
        
        fig.supxlabel(self._xlabel)
        plt.subplots_adjust(hspace=0)
        plt.show()
        
    def plot_volumes(self):
        data = self._bin_data()
        
        fig, axes = plt.subplots(self._nrows, self._ncols, figsize=(15, 10), sharex=True)
        
        # If there is more than one plot, flatten the axes
        try:
            axes = axes.flatten()
        except:
            axes = [axes]
        
        for ax, d, (z_start, z_end) in zip(axes, data, self._z_bins):
            counts, volumes, lum_bin_centers, _ = d
            
            # Mask the data if the number of galaxies in a luminosity bin is less than min_count
            good_mask = counts > self._min_count
            good_volumes = volumes[good_mask]
            good_lum_bin_centers = lum_bin_centers[good_mask]
            
            # Get the volumes for each luminosity bin not properly masked
            bad_mask = counts <= self._min_count
            bad_volumes = volumes[bad_mask]
            bad_lum_bin_centers = lum_bin_centers[bad_mask]
            
            ax.scatter(good_lum_bin_centers, good_volumes, label=f'{z_start} $\leq$ z < {z_end}')
            ax.scatter(bad_lum_bin_centers, bad_volumes, color='red', label=f'Masked Bins')
            ax.axhline(y=0, color='black', linestyle='--', label='Zero Volume')
            # ax.set_yscale('symlog')
            ax.legend()
            
        fig.supylabel('Volume $Mpc^3$')
        fig.supxlabel(self._xlabel)
        plt.subplots_adjust(hspace=0)
        plt.show()
    
if __name__ == '__main__':
    import pandas as pd
    
    # Read in CDFS main data
    data = pd.read_csv('./DATA/ZFOURGE/CDFS/CDFS_MAIN.csv')
    df = pd.DataFrame(data)
    df = df[df['Use'] == 1]
    df = df[df['FKs'] >= 0]
    df = df[df['FKs'] <= 27]
    df = df[df['SNR'] >= 6]
        
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # cosmology
    redshift_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)] # redshift bins
    lum_bins = np.arange(-25, -10, 0.5) # luminosity bins
    z = df['zpk'] # redshift
    m_app = df['FKs'] # apparent magnitude
    
    dists = cosmo.luminosity_distance(z) # Mpc
    dists *= 10 ** 6 # Mpc -> pc
    m_ab = 25 - 2.5*np.log10(m_app) # apparent magnitude -> AB magnitude 
    M_abs = m_ab - 5 * np.log10(dists / 10) # AB magnitude -> absolute magnitude s
    
    mlim = 27 # apparent magnitude limit
    survey_area = 0.03556 # survey area in square degrees
    
    # xlabel = '$M_{AB}$'
    # xlabel = 'log($L_{IR}$ [$L_{\odot}$])'
    
    lf = LF(cosmo, z, M_abs, redshift_bins, lum_bins, mlim, survey_area, min_count=10, ylim=(-10.5,-2))
    # lf.print_counts()
    # lf.print_volumes()
    # lf.plot()
    # lf.overlay_plot()
    # lf.plot_histograms()
    # lf.plot_volumes()
    lf.fit(func='Saunders', verbose=True, maxfev=100000)