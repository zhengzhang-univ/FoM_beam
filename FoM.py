import numpy as np
from scipy.fftpack import dctn, fftn, fftshift, fftfreq
from scipy.io import loadmat
from scipy.interpolate import griddata

from cora.signal import corr21cm
from cora.foreground import gaussianfg, galaxy
from cora.util import units

from mpiutil import parallel_map


def rad2deg(a):
    return a * 180. / np.pi

def deg2rad(a):
    return a * np.pi / 180.

class PointSources(gaussianfg.PointSources):
    """Scale up point source amplitude to a higher S_{cut} = 0.1 Jy"""
    A = 3.55e-5
    nu_0 = 408.0
    l_0 = 100.0


class FoM:
    pol_frac = 1.0
    tsys_flat = 50.0
    ndays = 365.0

    def __init__(self, nu_min, nu_max, ndim, beam_file_path, polarized=False):
        self.frequencies = np.linspace(nu_min, nu_max, ndim)
        # normalized frequency coordinates
        self.xs = (self.frequencies - nu_min) / (nu_max - nu_min)
        # Polarization version or not
        self.polarized = polarized
        self.frequency_window = np.sin(np.pi*self.xs)**4
        self.beam_file_path = beam_file_path
        self.spatial_treatments()
        self.fft()
        print("The Beam FoM object has been initialized! \n")

    def spatial_treatments(self):
        theta_max = deg2rad(75)
        E1 = loadmat(self.beam_file_path+'E1_S.mat')['E1_S']
        self.nfreq = E1.shape[-1]
        assert self.nfreq == len(self.frequencies)
        Valind = loadmat(self.beam_file_path+'ValInd.mat')['valInd'].flatten()
        theta = loadmat(self.beam_file_path+'th.mat')['th'].flatten()[Valind] # in radians
        phi = loadmat(self.beam_file_path+'ph.mat')['ph'].flatten()[Valind] # in radians
        radius = 2.*np.tan(theta_max/2.)
        # Interpolate:
        x, y = self.sphere2plane(theta,phi)
        points = np.column_stack((x,y))
        grid_x, grid_y = np.mgrid[-radius:radius:301j, -radius:radius:301j]
        E1 = self.interpolation(points, E1, grid_x, grid_y)
        theta_coord, phi_coord = self.plane2sphere(grid_x.flatten(), grid_y.flatten())
        theta_coord = theta_coord.reshape(grid_x.shape)
        E1 = self.directional_window(E1, theta_coord, theta_max)
        self.beam_intensity = self.field2scaledintensity(E1, theta_coord)
        E1 = loadmat(self.beam_file_path+'E1_S_Error.mat')['E1_S']
        E1 = self.interpolation(points, E1, grid_x, grid_y)
        E1 = self.directional_window(E1, theta_coord, theta_max)
        self.beam_err_intensity = self.field2scaledintensity(E1, theta_coord)
        self.x_coord, self.y_coord = np.unique(grid_x), np.unique(grid_y)

    def field2scaledintensity(self, E_field, theta_coord):
        intensity_rescaling_factor = np.cos(theta_coord) ** 5 / (np.cos(theta_coord) - np.sin(theta_coord))
        beam_intensity = intensity_rescaling_factor[:, :, np.newaxis] * (1 / 376.730313668) * np.abs(E_field) ** 2
        return beam_intensity

    def sphere2plane(self, theta, phi):
        x = 2. * np.sin(theta) * np.cos(phi) / (1. + np.cos(theta))
        y = 2. * np.sin(theta) * np.sin(phi) / (1. + np.cos(theta))
        return x,y

    def plane2sphere(self, x, y):
        phi = np.zeros(x.shape)
        for i in np.arange(x.shape[0]):
            if x[i] > 0.:
                phi[i] = np.arctan(y[i] / x[i])
            elif x[i] < 0.:
                phi[i] = np.arctan(y[i] / x[i]) + np.pi
            else:
                if y[i] > 0.:
                    phi[i] = np.pi / 2.
                elif y[i] < 0.:
                    phi[i] = - np.pi / 2.
        theta = 2.*np.arctan(np.sqrt(x**2+y**2)/2.)
        return theta, phi

    def directional_window(self, E_field, theta_coords, theta_max=75., alpha=0.05):
        for i in range(theta_coords.shape[0]):
            for j in range(theta_coords.shape[1]):
                if theta_coords[i,j] < theta_max:
                    E_field[i, j, :] *= np.exp(-alpha*(1/(theta_coords[i,j]-theta_max)**2 - 1/theta_max**2))
                else:
                    E_field[i, j, :] = 0.
        return E_field

    def interpolation(self, points, E_field, grid_x, grid_y):
        x, y = grid_x.shape
        E_interpolated = np.zeros(shape=(x, y, self.nfreq))
        for nu in range(self.nfreq):
            E_interpolated[:, :, nu] = griddata(points, E_field[:, nu], (grid_x, grid_y), method='cubic')
        return E_interpolated

    def fft(self):
        # frequency FT coordinates
        self.frequency_FT_coords = fftshift(fftfreq(self.frequencies.size, d=self.frequencies[1] - self.frequencies[0]))


        # Beam pixel resolutions
        x_res = np.abs(self.x_coord[1] - self.x_coord[0])
        y_res = np.abs(self.y_coord[1] - self.y_coord[0])
        # Projected beam coordinates in Fourier space (x_fft, y_fft) = (ell_x, ell_y)
        self.x_fft_coords = fftshift(fftfreq(self.x_coord.size, d=x_res))
        self.y_fft_coords = fftshift(fftfreq(self.y_coord.size, d=y_res))

        Beam_fft = fftn(self.beam_intensity, axes=(0, 1))
        Beam_err_fft = fftn(self.beam_err_intensity, axes=(0, 1))
        self.Beam_fft_shift = fftshift(Beam_fft, axes=(0, 1))
        Beam_err_fft_shift = fftshift(Beam_err_fft, axes=(0, 1))
        self.fractional_beam_err_k = Beam_err_fft_shift / self.Beam_fft_shift
        # Grid version
        y_fft_grid, x_fft_grid = np.meshgrid(self.x_fft_coords, self.y_fft_coords)
        self.k_perp_array = x_fft_grid +1j*y_fft_grid

    def k_to_l_flatsky(self, k_perp):
        return 2 * np.pi * np.absolute(k_perp)

    def clarray_21cm(self, k_perp):
        cr = corr21cm.Corr21cm()
        l = self.k_to_l_flatsky(k_perp)
        ct = cr.angular_powerspectrum(l, self.frequencies[:, np.newaxis], self.frequencies[np.newaxis, :])
        if not self.polarized:
            return ct
        else:
            nfreq = self.frequencies.size
            cv_fg = np.zeros((4, 4, nfreq, nfreq))
            cv_fg[0, 0] = ct
            return cv_fg

    def clarray_fg(self, k_perp):
        l = self.k_to_l_flatsky(k_perp)
        fsyn = galaxy.FullSkySynchrotron()
        fps = PointSources()
        if self.polarized:
            nfreq = self.frequencies.size
            cv_fg = np.zeros((4, 4, nfreq, nfreq))
            cv_fg[0, 0] = fsyn.angular_powerspectrum(l, self.frequencies[:, np.newaxis], self.frequencies[np.newaxis, :])
            fpol = galaxy.FullSkyPolarisedSynchrotron()
            cv_fg[1, 1] = self.pol_frac * fpol.angular_powerspectrum(l, self.frequencies[:, np.newaxis],
                                                                     self.frequencies[np.newaxis, :])
            cv_fg[2, 2] = self.pol_frac * fpol.angular_powerspectrum(l, self.frequencies[:, np.newaxis],
                                                                     self.frequencies[np.newaxis, :])
            cv_fg[0, 0] += fps.angular_powerspectrum(l, self.frequencies[:, np.newaxis], self.frequencies[np.newaxis, :])
        else:
            cv_fg = fsyn.angular_powerspectrum(l, self.frequencies[:, np.newaxis], self.frequencies[np.newaxis, :]) \
                  + fps.angular_powerspectrum(l, self.frequencies[:, np.newaxis], self.frequencies[np.newaxis, :])
        return cv_fg

    def t_sys(self, freq):
        return np.ones_like(freq) * self.tsys_flat

    def ps_noise(self):
        # In k_perp and frequency coordinates
        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6
        T_sys = self.t_sys(self.frequencies)
        xsize = self.x_fft_coords.size
        ysize = self.y_fft_coords.size
        fsize = self.frequencies.size
        result = np.zeros((xsize, ysize, fsize, fsize))
        for i in range(xsize):
            for j in range(ysize):
                result[i, j, :, :] = np.diag( (T_sys/np.abs(self.Beam_fft_shift[i,j])) ** 2
                                              / (units.t_sidereal * self.ndays * bw))
        return result

    def ps_noise_ij(self, i, j):
        # In k_perp and frequency coordinates
        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6
        T_sys = self.t_sys(self.frequencies)
        return np.diag((T_sys/np.abs(self.Beam_fft_shift[i, j])) ** 2 / (units.t_sidereal * self.ndays * bw))

    def spectral_window(self, data_array):
        sin_x = np.sin(self.xs * np.pi)
        W = np.outer(sin_x, sin_x)
        return data_array * W

    def Dct(self, data_array):
        return dctn(data_array, axes = (-2,-1))

    def P_ab_at_a_k_perp(self, i, j, SNR_only = False, include_1st_order = True):
        k_perp = self.x_fft_coords[i] + 1j * self.y_fft_coords[j]
        cl_21 = self.clarray_21cm(k_perp)
        P_ab_21 = self.Dct(self.spectral_window(cl_21))
        cl_fg = self.clarray_fg(k_perp)
        P_ab_fg = self.Dct(self.spectral_window(cl_fg))
        x2_grid, x1_grid = np.meshgrid(self.fractional_beam_err_k[i, j].conj(), self.fractional_beam_err_k[i, j])
        if include_1st_order:
            aux = (x2_grid + x1_grid + x2_grid * x1_grid) * (cl_21 + cl_fg)
        else:
            aux = x2_grid * x1_grid * (cl_21 + cl_fg)
        P_ab_beam = self.Dct(self.spectral_window(aux))
        x2_grid, x1_grid = np.meshgrid(self.Beam_fft_shift[i, j].conj(), self.Beam_fft_shift[i, j])
        aux = self.ps_noise_ij(i, j) / (x2_grid * x1_grid)
        P_ab_noise = self.Dct(self.spectral_window(aux))
        SNR = np.sum(np.abs(P_ab_21/(P_ab_fg + P_ab_beam + P_ab_noise)))
        if SNR_only:
            return SNR
        else:
            return P_ab_21, P_ab_fg, P_ab_beam, P_ab_noise, SNR

    def SNR_at_a_k_perp(self, i, j, include_1st_order = False):
        k_perp = self.x_fft_coords[i] + 1j * self.y_fft_coords[j]
        cl_21 = self.clarray_21cm(k_perp)
        P_ab_21 = self.Dct(self.spectral_window(cl_21))
        cl_noise = self.clarray_fg(k_perp).astype(complex)
        x2_grid, x1_grid = np.meshgrid(self.fractional_beam_err_k[i, j].conj(), self.fractional_beam_err_k[i, j])
        if include_1st_order:
            cl_noise += (x2_grid + x1_grid + x2_grid * x1_grid) * (cl_21 + cl_noise)
        else:
            cl_noise += (x2_grid * x1_grid) * (cl_21 + cl_noise)
        x2_grid, x1_grid = np.meshgrid(self.Beam_fft_shift[i, j].conj(), self.Beam_fft_shift[i, j])
        cl_noise += self.ps_noise_ij(i, j) / (x2_grid * x1_grid)
        P_ab_noise = self.Dct(self.spectral_window(cl_noise))
        return np.sum(np.abs(P_ab_21/P_ab_noise))

    def FoM(self, include_1st_order = False):
        xsize = self.x_fft_coords.size
        ysize = self.y_fft_coords.size
        def func(i):
            return self.SNR_at_a_k_perp(i, i, include_1st_order)
        # for i in np.arange(xsize):
        aux = parallel_map(func, list(np.arange(ysize)))
        FoM = sum(aux)
        return FoM








