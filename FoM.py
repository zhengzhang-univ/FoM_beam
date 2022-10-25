import numpy as np
from scipy.fftpack import dctn, fftn, fftshift, fftfreq
from scipy.io import loadmat
from scipy.interpolate import RectBivariateSpline

from cora.signal import corr21cm
from cora.foreground import gaussianfg, galaxy
from cora.util import units

from mpiutil import parallel_map


def sphere2plane(theta, phi):
    x = 2. * np.sin(theta) * np.cos(phi) / (1. + np.cos(theta))
    y = 2. * np.sin(theta) * np.sin(phi) / (1. + np.cos(theta))
    return x, y


def plane2sphere(x, y):
    # Function mapping the flat-sky coordinates to the spherical coordinates.
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
    theta = 2. * np.arctan(np.sqrt(x ** 2 + y ** 2) / 2.)
    return phi, theta


def load_beam_files(beam_file_path):
    E1 = loadmat(beam_file_path + 'E1_S.mat')['E1_S']
    E1_err = loadmat(beam_file_path + 'E1_S_Error.mat')['E1_S']
    valind = loadmat(beam_file_path + 'ValInd.mat')['valInd'].flatten()
    theta = loadmat(beam_file_path + 'th.mat')['th'].flatten()[valind]
    phi = loadmat(beam_file_path + 'ph.mat')['ph'].flatten()[valind]
    valind = np.where(np.rad2deg(theta) < 90)[0]
    E1 = E1[valind]
    E1_err = E1_err[valind]
    theta = theta[valind]
    phi = phi[valind]
    uni_phi, uni_theta = np.unique(phi), np.unique(theta)
    nphi, ntheta =len(uni_phi), len(uni_theta)
    return E1.reshape(nphi, ntheta, -1), E1_err.reshape(nphi, ntheta, -1), \
           uni_phi, uni_theta


class PointSources(gaussianfg.PointSources):
    """Scale up point source amplitude to a higher S_{cut} = 0.1 Jy"""
    A = 3.55e-5
    nu_0 = 408.0
    l_0 = 100.0


class FoM:
    pol_frac = 1.0
    tsys_flat = 50.0
    d_dish = 6.0  # meter
    eta = 0.75 # dish efficiency
    ndays = 730.0
    survey_area = 15000.0
    FOV = 0.0076    # field of view: 25.0 square degree.
    num_beams = 2*1024
    num_pol = 2
    num_dish = 1024
    noiseps_factor = (4/(eta*np.pi))**2 * (survey_area/25.0)/(num_pol*num_dish)

    def __init__(self, nu_min, nu_max, ndim, beam_file_path, polarized=False):
        self.frequencies = np.linspace(nu_min, nu_max, ndim)
        # normalized frequency coordinates
        self.xs = (self.frequencies - nu_min) / (nu_max - nu_min)
        # Polarization version or not
        self.polarized = polarized
        self.frequency_window = np.sin(np.pi*self.xs)**4 # frequency window function
        self.beam_file_path = beam_file_path
        self.spatial_treatments()
        print("The spatial treatments are done! \n")
        self.fft()
        print("FFT is done! \n")
        print("The Beam FoM object has been initialized! \n")

    def spatial_treatments(self, Ndim=301, theta_max=75):
        # This function perform all spatial treatments on the beam, including
        #       1. map spherical coordinates to (x,y) coordinates on the flat sky
        #       2. Define a (x,y) grid.  Interpolate the E field on it.
        #       3. Generate the beam intensity and rescale it to fit the flat sky approximation.
        #       4. Apply the spatial window function to the beam pattern.
        self.Ndim = Ndim
        E1, E1_err, phi, theta = load_beam_files(self.beam_file_path)
        self.nfreq = E1.shape[-1]
        assert self.nfreq == len(self.frequencies)

        thetaMax = np.deg2rad(theta_max)
        radius = 2.*np.tan(thetaMax/2.)
        self.x_coord, self.y_coord = np.linspace(-radius, radius, Ndim), np.linspace(-radius, radius, Ndim)
        grid_y, grid_x = np.meshgrid(self.x_coord, self.y_coord)
        target_phi, target_theta = plane2sphere(grid_x.flatten(), grid_y.flatten())

        # Interpolation:
        E1 = self.interpolation(phi, theta, E1, target_phi, target_theta)
        # Rescaling power density for the projected field
        grid_target_theta = target_theta.reshape(grid_x.shape)
        beam_intensity = self.field2scaledintensity(E1, grid_target_theta)
        # Normalize the beam
        normalization_factor = self.nfreq / np.linalg.norm(beam_intensity)
        beam_intensity *= normalization_factor
        # Apply the directional window
        self.beam_intensity = self.directional_window(beam_intensity, grid_target_theta, theta_max) # apply the directional window

        # interpolate, rescale, and apodize the beam error.
        E1_err = self.interpolation(phi, theta, E1_err, target_phi, target_theta)
        beam_err_intensity = self.field2scaledintensity(E1_err, grid_target_theta) * normalization_factor
        self.beam_err_intensity = self.directional_window(beam_err_intensity, grid_target_theta, theta_max)
        return

    def field2scaledintensity(self, E_field, theta_coord):
        aux_theta = theta_coord/2.
        intensity_rescaling_factor = np.cos(aux_theta) ** 5 / (np.cos(aux_theta) - np.sin(aux_theta))
        beam_intensity = intensity_rescaling_factor[:, :, np.newaxis] * np.abs(E_field) ** 2
        return beam_intensity

    def directional_window(self, E_field, theta_coords, theta_max=75., alpha=0.05):
        for i in range(theta_coords.shape[0]):
            for j in range(theta_coords.shape[1]):
                if theta_coords[i,j] < theta_max:
                    E_field[i, j, :] *= np.exp(-alpha*(1/(theta_coords[i,j]-theta_max)**2 - 1/theta_max**2))
                else:
                    E_field[i, j, :] = 0.
        return E_field

    def interpolation(self, phi, theta, E_field, target_phi, target_theta):
        E_interpolated = np.zeros(shape=(self.Ndim, self.Ndim, self.nfreq))
        for nu in range(self.nfreq):
            interp = RectBivariateSpline(phi, theta, E_field[:, :, nu])
            E_interpolated[:, :, nu] = interp(target_phi, target_theta, grid=False).reshape(self.Ndim, self.Ndim)
        return E_interpolated

    def fft(self):
        # This is the function performing FFT on the beam and beam error over spatial degrees of freedom.

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
        # y_fft_grid, x_fft_grid = np.meshgrid(self.x_fft_coords, self.y_fft_coords)
        # self.k_perp_array = x_fft_grid +1j*y_fft_grid

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

    def ps_noise_ij(self, i, j):
        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6
        T_sys = self.t_sys(self.frequencies)
        return np.diag((T_sys / np.abs(self.Beam_fft_shift[i, j])) ** 2
                        / (units.t_sidereal * self.ndays * bw)) * self.noiseps_factor

    def spectral_window(self, data_array):
        sin_x = np.sin(self.xs * np.pi)
        W = np.outer(sin_x, sin_x)
        return data_array * W

    def Dct(self, data_array):
        return dctn(data_array, axes=(-2, -1))

    def P_ab_at_a_k_perp(self, i, j):
        # Forbidden case: x_fft_coords[i] = y_fft_coords[j] = 0
        k_perp = self.x_fft_coords[i] + 1j * self.y_fft_coords[j]
        cl_21 = self.clarray_21cm(k_perp)
        P_ab_21 = self.Dct(self.spectral_window(cl_21))
        if k_perp == 0.:
            P_ab_21 = np.zeros(shape=P_ab_21.shape)
            P_ab_fg, P_ab_beam, P_ab_noise = np.ones((3,)+P_ab_21.shape)
        else:
            cl_fg = self.clarray_fg(k_perp)
            P_ab_fg = self.Dct(self.spectral_window(cl_fg))
            x2_grid, x1_grid = np.meshgrid(self.fractional_beam_err_k[i, j].conj(), self.fractional_beam_err_k[i, j])
            aux = x2_grid * x1_grid * (cl_21 + cl_fg)
            P_ab_beam = self.Dct(self.spectral_window(aux))
            x2_grid, x1_grid = np.meshgrid(self.Beam_fft_shift[i, j].conj(), self.Beam_fft_shift[i, j])
            # aux = self.ps_noise_ij(i, j) / (x2_grid * x1_grid)
            aux = self.ps_noise_ij(i, j)
            P_ab_noise = self.Dct(self.spectral_window(aux))
        return P_ab_21, P_ab_fg, P_ab_beam, P_ab_noise

    def SNR_at_a_k_perp(self, i, j):
        P_ab_21, P_ab_fg, P_ab_beam, P_ab_noise = self.P_ab_at_a_k_perp(i, j)
        SNR = np.sum(np.diag(P_ab_21)/np.diag(P_ab_fg + P_ab_noise).real)
        SNR_bar = np.sum(np.diag(P_ab_21)/np.diag(P_ab_fg + P_ab_beam + P_ab_noise).real)
        return SNR, SNR_bar

    def FoM(self):
        xsize = self.x_fft_coords.size
        ysize = self.y_fft_coords.size
        SNR_array = np.zeros((xsize, ysize, 2))
        for i in np.arange(xsize):
            def func(j):
                return self.SNR_at_a_k_perp(i, j)
            SNR_array[i, :, :] = np.array(parallel_map(func, list(np.arange(ysize))))
        return np.sum(SNR_array[:, :, 0])/np.sum(SNR_array[:, :, 1])

    def SNR_at_a_k_perp_v2(self, i, j):
        P_ab_21, P_ab_fg, P_ab_beam, P_ab_noise = self.P_ab_at_a_k_perp(i, j)
        SNR = np.abs(P_ab_21/(P_ab_fg + P_ab_noise)).sum()
        SNR_bar = np.abs(P_ab_21/(P_ab_fg + P_ab_beam + P_ab_noise)).sum()
        return SNR, SNR_bar
    def FoM_v2(self):
        xsize = self.x_fft_coords.size
        ysize = self.y_fft_coords.size
        SNR_array = np.zeros((xsize, ysize, 2))
        for i in np.arange(xsize):
            def func(j):
                return self.SNR_at_a_k_perp_v2(i, j)
            SNR_array[i, :, :] = np.array(parallel_map(func, list(np.arange(ysize))))
        return np.sum(SNR_array[:, :, 0])/np.sum(SNR_array[:, :, 1])

