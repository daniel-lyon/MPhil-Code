import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.cosmology import FlatLambdaCDM

class LF(object):
    def __init__(self, cosmo, z, M_abs, z_bins, lum_bins, mlim, survey_area):
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
        """
        self._cosmo = cosmo
        self._z = z
        self._z_bins = z_bins
        self._lum_bins = lum_bins
        self._mlim = mlim
        self._survey_area = survey_area
          
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
            all_bins.append([counts, volumes, lum_bin_centers])
        return all_bins
    
    def _calc_volume(self, lum_bin_centers, min_z, max_z):
        """ 
        Calculate the volume of each luminosity bin.
        """
        # Calculate the minimum and maximum distance of the redshift bin
        dmin = self._cosmo.comoving_distance(min_z).value # Mpc
        dmax = self._cosmo.comoving_distance(max_z).value # Mpc

        # Calculate the maximum distance of each luminosity bin
        dmaxs = (10 * 10 ** ((self._mlim - lum_bin_centers) / 5)) # pc
        dmaxs = dmaxs / 10 ** 6 # pc -> Mpc

        # If the maximum distance is greater than the maximum redshift bin distance, set it to the maximum redshift bin distance
        dmaxs[dmaxs > dmax] = dmax

        # Calculate the minimum volume of the redshift bin
        vmin = 4/3 * np.pi * dmin**3 # Mpc^3

        # Calculate the maximum volume of each luminosity bin
        vmaxs = 4/3 * np.pi * dmaxs**3 #

        # Total volume probed accounting for survey area
        vol = (vmaxs - vmin) * (self._survey_area / 41253) # Mpc^3
        return vol
    
    def plot(self, min_count=0):
        """ 
        Make a simple plot of the luminosity function with no fitting.
        
        Parameters
        ----------
        min_count : int, optional
            Minimum number of galaxies in a luminosity bin to be plotted. Default is 0.
        """
        data = self._bin_data()
        _, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
        
        # Plot the luminosity function for each redshift bin
        for ax, d, (z_start, z_end) in zip(axes.flatten(), data, self._z_bins):
            counts, volumes, lum_bin_centers = d
            
            # Mask the data if the number of galaxies in a luminosity bin is less than min_count
            mask = counts > min_count
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
        return phi_star * 10 ** (-0.4*(alpha+1)*(M-M_star)) * np.exp(-10.**(-0.4*(M-M_star)))
    
    @staticmethod
    def _saunders_magnitude(M, M_star, phi_star, alpha, sigma):
        return phi_star * 10 ** (-0.4*(1-alpha)*(M-M_star)) * np.exp(-1 / (2 * sigma ** 2) * (np.log10(1 + 10 ** (0.4 * (M_star - M))))**2)
        
    def _get_params(self, func, lum_bin_centers, lf, z_start, z_end, maxfev=1000, p0=None):
        """ 
        Get the parameters of the function fit.
        """
        if p0 is None and func == 'Schechter':
            p0 = [lum_bin_centers[0], 0.001, -0.9]
            f = self._schechter_magnitude
        elif p0 is None and func == 'Saunders':
            p0 = [lum_bin_centers[0], 0.001, -0.9, 0.1]
            f = self._saunders_magnitude
        
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
        
    def fit(self, func, min_count=0, maxfev=1000, p0=None, nrows=3, ncols=2, verbose=True):
        """ 
        Fit a given function to the luminosity function.
        
        Parameters
        ----------
        func : str
            The function to fit. Options are 'Schechter' or 'Saunders'.
        
        min_count : int, optional
            Minimum number of galaxies in a luminosity bin to be plotted. Default is 0.
        
        maxfev : int, optional
            Maximum number of calls to function. Default is 1000.
        
        p0 : array-like, optional
            Initial guess for the parameters. Default 'None' will auto set.
        
        nrows : int, optional
            Number of rows in the plot. Default is 3.
        
        ncols : int, optional
            Number of columns in the plot. Default is 2.
            
        verbose : bool, optional
            Print the fit parameters. Default is True.
        """
        data = self._bin_data()
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10), sharex=True, sharey=True)

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
        
        # For each redshift bin, fit the function to the data
        for ax, d, (z_start, z_end) in zip(axes, data, self._z_bins):
            counts, volumes, lum_bin_centers = d
            
            # Mask the data if the number of galaxies in a luminosity bin is less than min_count
            mask = counts > min_count
            lf = counts[mask] / volumes[mask]
            lum_bin_centers = lum_bin_centers[mask]
            
            # Calculate the error in the luminosity function
            upp_lf, low_lf = self._phi_error(counts[mask], volumes[mask])
            
            # Fit the function to the data
            params = self._get_params(func, lum_bin_centers, lf, z_start, z_end, maxfev, p0)
            if params is False:
                continue
            upper_params = self._get_params(func, lum_bin_centers, upp_lf, z_start, z_end, maxfev, p0)
            lower_params = self._get_params(func, lum_bin_centers, low_lf, z_start, z_end, maxfev, p0)
            
            # Print the fit parameters
            if verbose:
                print(f'{z_start} <= z < {z_end} Function fit:')
                print(f'M_star = {params[0]:.2e}')
                print(f'phi_star = {params[1]:.2e}')
                print(f'alpha = {params[2]:.2e}')
                if len(params) == 4:
                    print(f'sigma = {params[3]:.2e}')
                print('\n')
            
            # Plot the luminosity function and the fit
            ax.errorbar(lum_bin_centers, np.log10(lf), yerr=[np.log10(lf) - np.log10(low_lf), np.log10(upp_lf) - np.log10(lf)], 
                fmt='o', label=f'{z_start} $\leq$ z < {z_end}', capsize=4)
            ax.plot(lum_bin_centers, np.log10(f(lum_bin_centers, *params)), label=f'{func} fit', color='red', linestyle='--')
            long_lum_smooth = np.linspace(lum_bin_centers[0], lum_bin_centers[-1], 100)
            ax.fill_between(long_lum_smooth, 
                np.log10(f(long_lum_smooth, *lower_params)), 
                np.log10(f(long_lum_smooth, *upper_params)), 
                alpha=0.2, color='red', label='Fit Error')
            # ax.fill_between(lum_bin_centers, np.log10(low_lf), np.log10(upp_lf), alpha=0.2, color='magenta', label='Error')
            ax.set_ylim(-5.25, -2.25)
            ax.legend(loc='lower right')

        fig.suptitle('ZFOURGE Daniel Code')
        fig.supylabel('log($\phi$ $Mpc^-3$)')
        fig.supxlabel('$M_{AB}$')
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
            counts, _, _ = d
            print(f'{z_start} <= z < {z_end}: {counts.tolist()}. Total = {np.sum(counts)}')
        print('\n')
        return counts
    
    def overlay_plot(self, min_count=0):
        """ 
        Make a simple plot of the luminosity function with no fitting.
        
        Parameters
        ----------
        min_count : int, optional
            Minimum number of galaxies in a luminosity bin to be plotted. Default is 0.
        """
        data = self._bin_data()
        
        # Plot the luminosity function for each redshift bin
        for d, (z_start, z_end) in zip(data, self._z_bins):
            counts, volumes, lum_bin_centers = d
            
            # Mask the data if the number of galaxies in a luminosity bin is less than min_count
            mask = counts > min_count
            lf = counts[mask] / volumes[mask]
            lum_bin_centers = lum_bin_centers[mask]
            
            # Plot the luminosity function
            plt.plot(lum_bin_centers, np.log10(lf), label=f'{z_start} $\leq$ z < {z_end}')
        
        plt.legend()
        plt.tight_layout()
        plt.xlabel('$M_{AB}$')
        plt.ylabel('log(phi)')
        plt.show()
    
if __name__ == '__main__':
    import pandas as pd
    
    # Read in CDFS main data
    data = pd.read_csv('./DATA/ZFOURGE/CDFS/CDFS_MAIN.csv')
    df = pd.DataFrame(data)
    df = df[df['Use'] == 1]
    df = df[df['FKs'] >= 0]
        
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3) # cosmology
    redshift_bins = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)] # redshift bins
    lum_bins = np.arange(-25, -10, 0.5) # luminosity bins
    z = df['zpk'] # redshift
    m_app = df['FKs'] # apparent magnitude
    
    dists = cosmo.luminosity_distance(z) # Mpc
    dists *= 10 ** 6 # Mpc -> pc
    m_ab = 25 - 2.5*np.log10(m_app) # apparent magnitude -> AB magnitude 
    M_abs = m_ab - 5 * np.log10(dists / 10) # AB magnitude -> absolute magnitude 
    
    mlim = 27 # apparent magnitude limit
    survey_area = 0.03556 # survey area in square degrees
    
    lf = LF(cosmo, z, M_abs, redshift_bins, lum_bins, mlim, survey_area)
    lf.print_counts()
    # lf.plot(min_count=10)
    lf.fit(func='Schechter', min_count=10)
    # lf.overlay_plot(min_count=10)