
import numpy as np
from scipy.fft import dctn, fftn, fftshift, fftfreq
from scipy.io import loadmat

from cora.signal import corr21cm
from cora.foreground import gaussianfg, galaxy
from cora.util import units


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
        # Beam Intensity
        self.BeamPath = beam_file_path
        self.Beam = loadmat('{:s}/u_freq_cube_hirax.mat'.format(self.BeamPath))['U_FreqCube']
        self.BeamErr = self.Beam*0.2
        self.fft_coordinates()

    def fft_coordinates(self):
        # frequency FT coordinates
        self.frequency_FT_coords = fftshift(fftfreq(self.frequencies.size, d=self.frequencies[1] - self.frequencies[0]))

        # Beam Coordinates
        xgrid = loadmat('{:s}/XXint_hirax.mat'.format(self.BeamPath))['XXint']
        ygrid = loadmat('{:s}/YYint_hirax.mat'.format(self.BeamPath))['YYint']
        # Projected beam coordinates in image space (x, y) = (theta_x, theta_y) in radians
        x_coords = np.unique(xgrid)
        y_coords = np.unique(ygrid)
        # Beam pixel resolutions
        x_res = np.abs(x_coords[1] - x_coords[0])
        y_res = np.abs(y_coords[1] - y_coords[0])
        # Projected beam coordinates in Fourier space (x_fft, y_fft) = (ell_x, ell_y)
        self.x_fft_coords = fftshift(fftfreq(x_coords.size, d=x_res))
        self.y_fft_coords = fftshift(fftfreq(y_coords.size, d=y_res))

        Beam_fft = fftn(self.Beam, axes=(0, 1))
        Beam_err_fft = fftn(self.BeamErr, axes=(0, 1))
        self.Beam_fft_shift = fftshift(Beam_fft, axes=(0, 1))
        Beam_err_fft_shift = fftshift(Beam_err_fft, axes=(0, 1))
        self.Fractional_beam_err = Beam_err_fft_shift / self.Beam_fft_shift
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


    def windowing(self, data_array):
        sin_x = np.sin(self.xs * np.pi)
        W = np.outer(sin_x, sin_x)
        return data_array * W

    def Dct(self, data_array):
        return dctn(data_array, axes = (-2,-1) )

    def P_ab_at_a_k_perp(self, i, j, SNR_only = False):
        k_perp = self.x_fft_coords[i] + 1j * self.y_fft_coords[j]
        cl_21 = self.clarray_21cm(k_perp)
        P_ab_21 = self.Dct(self.windowing(cl_21))
        cl_fg = self.clarray_fg(k_perp)
        P_ab_fg = self.Dct(self.windowing(cl_fg))
        x2_grid, x1_grid = np.meshgrid(self.Fractional_beam_err[i, j].conj(), self.Fractional_beam_err[i, j])
        aux = (x2_grid + x1_grid + x2_grid * x1_grid) * (cl_21 + cl_fg)
        P_ab_beam = self.Dct(self.windowing(aux))
        x2_grid, x1_grid = np.meshgrid(self.Beam_fft_shift[i, j].conj(), self.Beam_fft_shift[i, j])
        aux = self.ps_noise_ij(i, j) / (x2_grid * x1_grid)
        P_ab_noise = self.Dct(self.windowing(aux))
        SNR = np.sum(np.abs(P_ab_21/(P_ab_fg + P_ab_beam + P_ab_noise)))
        if SNR_only:
            return SNR
        else:
            return P_ab_21, P_ab_fg, P_ab_beam, P_ab_noise, SNR

    def SNR_at_a_k_perp(self, i, j):
        k_perp = self.x_fft_coords[i] + 1j * self.y_fft_coords[j]
        cl_21 = self.clarray_21cm(k_perp)
        P_ab_21 = self.Dct(self.windowing(cl_21))
        cl_noise = self.clarray_fg(k_perp).astype(complex)
        x2_grid, x1_grid = np.meshgrid(self.Fractional_beam_err[i, j].conj(), self.Fractional_beam_err[i, j])
        cl_noise += (x2_grid + x1_grid + x2_grid * x1_grid) * (cl_21 + cl_noise)
        x2_grid, x1_grid = np.meshgrid(self.Beam_fft_shift[i, j].conj(), self.Beam_fft_shift[i, j])
        cl_noise += self.ps_noise_ij(i, j) / (x2_grid * x1_grid)
        P_ab_noise = self.Dct(self.windowing(cl_noise))
        return np.sum(np.abs(P_ab_21/P_ab_noise))

    def FoM(self):
        xsize = self.x_fft_coords.size
        ysize = self.y_fft_coords.size
        SNR = 0.
        for i in np.arange(xsize):
            for j in np.arange(ysize):
                SNR += self.SNR_at_a_k_perp(i, j)
        return SNR








