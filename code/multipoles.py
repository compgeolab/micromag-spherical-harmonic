from harmonica._spherical_harmonics import legendre
import numpy as np
import scipy as sp
import pickle
import matplotlib.pyplot as plt
import xarray as xr
import harmonica as hm
import verde as vd
import verde.base
import numba
import choclo.constants


@numba.jit(parallel=True, nopython=True)
def spherical_harmonics_bz_jacobian(
    longitude,
    colatitude,
    normalized_radius,
    min_degree,
    max_degree,
    jacobian,
):
    n_data = longitude.size
    n_gs = sum(range(max_degree + 2)) - sum(range(min_degree + 1))
    for i in numba.prange(n_data):
        cos_colat = np.cos(colatitude[i])
        sin_colat = np.sin(colatitude[i])
        # Have to allocate here because of the parallel loop. These are small
        # for low degree so not a huge time sink.
        p = np.empty((max_degree + 1, max_degree + 1))
        p_deriv = np.empty_like(p)
        legendre.associated_legendre_schmidt(cos_colat, max_degree, p)
        legendre.associated_legendre_schmidt_derivative(max_degree, p, p_deriv)
        # Pre-compute the sin and cos of longitude to avoid repeated
        # computation for every value of n. Use the recursive Chebyshev method
        # to calculate sin/cos(m lon) to save running trig functions. This
        # method is about 20% faster than running the sin and cos several
        # times. See:
        # https://en.wikipedia.org/wiki/List_of_trigonometric_identities
        cos_mlon = np.empty(max_degree + 1)
        sin_mlon = np.empty(max_degree + 1)
        cos_mlon[0] = 1
        sin_mlon[0] = 0
        cos_mlon[1] = np.cos(longitude[i])
        sin_mlon[1] = np.sin(longitude[i])
        for m in range(2, max_degree + 1):
            cos_mlon[m] = 2 * cos_mlon[1] * cos_mlon[m - 1] - cos_mlon[m - 2]
            sin_mlon[m] = 2 * cos_mlon[1] * sin_mlon[m - 1] - sin_mlon[m - 2]
        # Calculating the power like this results in about an 8% performance
        # boost. Not much but I'll take it.
        r_frac = normalized_radius[i] ** (min_degree - 1 + 2)
        for n in range(min_degree, max_degree + 1):
            r_frac *= normalized_radius[i]
            for m in range(0, n + 1):
                jacobian[i, sum(range(n + 1)) + m - sum(range(min_degree + 1))] = r_frac * cos_mlon[m] * (
                    (n + 1) * cos_colat * p[n, m] + sin_colat * p_deriv[n, m]
                )
                if m > 0:
                    jacobian[i, n_gs + sum(range(n)) + m - 1 - sum(range(min_degree))] = r_frac * sin_mlon[m] * (
                        (n + 1) * cos_colat * p[n, m] + sin_colat * p_deriv[n, m]
                    )

@numba.jit(parallel=True, nopython=True)
def spherical_harmonics_bz(
    longitude,
    colatitude,
    normalized_radius,
    g,
    h,
    min_degree,
    max_degree,
    b_z,
):
    n_data = longitude.size
    for i in numba.prange(n_data):
        cos_colat = np.cos(colatitude[i])
        sin_colat = np.sin(colatitude[i])
        # Have to allocate here because of the parallel loop. These are small
        # for low degree so not a huge time sink.
        p = np.empty_like(g)
        p_deriv = np.empty_like(g)
        legendre.associated_legendre_schmidt(cos_colat, max_degree, p)
        legendre.associated_legendre_schmidt_derivative(max_degree, p, p_deriv)
        # Pre-compute the sin and cos of longitude to avoid repeated
        # computation for every value of n. Use the recursive Chebyshev method
        # to calculate sin/cos(m lon) to save running trig functions. This
        # method is about 20% faster than running the sin and cos several
        # times. See:
        # https://en.wikipedia.org/wiki/List_of_trigonometric_identities
        cos_mlon = np.empty(max_degree + 1)
        sin_mlon = np.empty(max_degree + 1)
        cos_mlon[0] = 1
        sin_mlon[0] = 0
        cos_mlon[1] = np.cos(longitude[i])
        sin_mlon[1] = np.sin(longitude[i])
        for m in range(2, max_degree + 1):
            cos_mlon[m] = 2 * cos_mlon[1] * cos_mlon[m - 1] - cos_mlon[m - 2]
            sin_mlon[m] = 2 * cos_mlon[1] * sin_mlon[m - 1] - sin_mlon[m - 2]
        # Calculating the power like this results in about an 8% performance
        # boost. Not much but I'll take it.
        r_frac = normalized_radius[i] ** (min_degree - 1 + 2)
        for n in range(min_degree, max_degree + 1):
            r_frac *= normalized_radius[i]
            for m in range(n + 1):
                b_z[i] += (
                    r_frac
                    * (g[n, m] * cos_mlon[m] + h[n, m] * sin_mlon[m])
                    * ((n + 1) * cos_colat * p[n, m] + sin_colat * p_deriv[n, m])
                )


class SphericalHarmonicsBz:

    def __init__(self, location, damping=1e-1, min_degree=1, max_degree=10):
        self.location = location
        self.damping = damping
        self.max_degree = max_degree
        self.min_degree = min_degree

    def fit(self, coordinates, data, weights=None):
        coordinates = tuple(np.atleast_1d(c).ravel() for c in coordinates)
        data = np.atleast_1d(data).ravel()
        x = coordinates[0] - self.location[0] 
        y = coordinates[1] - self.location[1] 
        z = coordinates[2] - self.location[2] 
        radius = np.sqrt(x**2 + y**2 + z**2)
        self.reference_radius_ = radius.min()
        colatitude = np.arccos(z / radius)
        longitude = np.arctan2(y, x)
        n_gs = sum(range(self.max_degree + 2)) - sum(range(self.min_degree + 1))
        n_hs = sum(range(self.max_degree + 1)) - sum(range(self.min_degree))
        n_params = n_gs + n_hs
        n_data = data.size
        jacobian = np.empty((n_data, n_params))
        spherical_harmonics_bz_jacobian(
            longitude,
            colatitude,
            self.reference_radius_ / radius,
            self.min_degree,
            self.max_degree,
            jacobian,
        )
        self.jacobian_ = jacobian
        parameters = vd.base.least_squares(jacobian, data, weights, damping=self.damping)
        # parameters = sp.linalg.lstsq(jacobian, data)[0]
        self.g_ = np.zeros((self.max_degree + 1, self.max_degree + 1))
        self.h_ = np.zeros((self.max_degree + 1, self.max_degree + 1))
        for n in range(self.min_degree, self.max_degree + 1):
            for m in range(0, n + 1):
                self.g_[n, m] = parameters[sum(range(n + 1)) + m - sum(range(self.min_degree + 1))]
                if m > 0:
                    self.h_[n, m] = parameters[n_gs + sum(range(n)) + m - 1 - sum(range(self.min_degree))]
        self.magnetic_moment_ = (
            4 * np.pi * (self.reference_radius_ * 1e-6)**3 
            * np.array([self.g_[1, 1], self.h_[1, 1], self.g_[1, 0]])
            * 1e-9
            / choclo.constants.VACUUM_MAGNETIC_PERMEABILITY
        )
        return self
        

    def predict(self, coordinates, min_degree=None, max_degree=None):
        if min_degree is None:
            min_degree = 1
        if max_degree is None:
            max_degree = self.max_degree
        cast = np.broadcast(*[np.atleast_1d(c) for c in coordinates])        
        coordinates = tuple(np.atleast_1d(c).ravel() for c in coordinates)
        x = coordinates[0] - self.location[0]
        y = coordinates[1] - self.location[1] 
        z = coordinates[2] - self.location[2]
        radius = np.sqrt(x**2 + y**2 + z**2)
        colatitude = np.arccos(z / radius)
        longitude = np.arctan2(y, x)
        result = np.zeros(cast.shape).ravel()
        spherical_harmonics_bz(
            longitude,
            colatitude,
            self.reference_radius_ / radius,
            self.g_,
            self.h_,
            min_degree,
            max_degree,
            result,
        )
        return result.reshape(cast.shape)


    # def update_location(self, coordinates, data):

    #     original_location = self.location.copy()

    #     if hasattr(data, "bz"):
    #         values = data.bz
    #     else:
    #         values = data

    #     def objective_function(z):

    #         location = original_location.copy()
    #         location[-1] = z[0]  # only the z coordinate

    #         self.location = location

    #         self.fit(coordinates, values)

    #         predicted = self.predict(coordinates, self.min_degree, self.max_degree)
    #         residuals = values - predicted.ravel()

    #         return np.sqrt(np.mean(residuals**2))

    #     result = sp.optimize.minimize(
    #         objective_function,
    #         x0=[self.location[-1]],
    #         method="Nelder-Mead",
    #         tol=1.0e-2,
    #         options={"maxiter": 500}
    #     )

    #     if result.success and np.isfinite(result.x[0]):
    #         self.location[-1] = result.x[0]
    #     else:
    #         self.location = original_location

    #     return self


    def update_location(self, coordinates, data, update_only_z=True):
        """
        Refine the source location by minimizing the RMS misfit between
        observed and predicted magnetic anomalies.
    
        Parameters
        ----------
        coordinates : tuple
            Coordinates of the observation points.
        data : xarray.DataArray, pandas.Series, or numpy.ndarray
            Observed magnetic field data. If the object has a ``bz`` attribute,
            it is used as the observed data.
        update_only_z : bool, default=True
            If True, only the vertical (z) coordinate is optimized. If False,
            all three coordinates (x, y, z) are optimized.
    
        Returns
        -------
        self : SphericalHarmonicsBz
            The model with the refined source location.
        """
    
        if hasattr(data, "bz"):
            values = data.bz
        else:
            values = data
    
        original_location = np.asarray(self.location, dtype=float).copy()
    
        if update_only_z:
            # Optimize only z. The initial value is normalized to unit magnitude
            # so that the Nelder-Mead simplex operates in a normalized space.
            scale = np.abs(original_location[-1])
    
            if scale == 0:
                scale = 1.0
    
            normalized_location = np.array(
                [original_location[-1] / scale]
            )
    
        else:
            # Normalize each coordinate independently so that x, y, and z
            # have comparable scales within the optimization simplex.
            scale = np.abs(original_location)
            scale[scale == 0] = 1.0
    
            normalized_location = original_location / scale
    
        def objective_function(location):
    
            if update_only_z:
                # Convert the normalized z coordinate back to physical units
                # while keeping x and y fixed at their original values.
                physical_location = original_location.copy()
                physical_location[-1] = location[0] * scale
    
            else:
                # Convert all normalized coordinates back to physical units.
                physical_location = location * scale
    
            self.location = physical_location
    
            # Refit the spherical-harmonic coefficients for each trial location.
            self.fit(coordinates, values)
    
            predicted = self.predict(
                coordinates,
                self.min_degree,
                self.max_degree
            )
    
            residuals = values - predicted.ravel()
            rms = np.sqrt(np.mean(residuals**2))
    
            return rms
    
        result = sp.optimize.minimize(
            objective_function,
            x0=normalized_location,
            method="Nelder-Mead",
            tol=1.0e-2,
            options={"maxiter": 100}
        )
    
        if result.success and np.all(np.isfinite(result.x)):
    
            if update_only_z:
                # Recover the optimized physical z coordinate while keeping
                # the original x and y coordinates unchanged.
                self.location = original_location.copy()
                self.location[-1] = result.x[0] * scale
    
            else:
                # Recover all optimized coordinates in physical units.
                self.location = result.x * scale
    
        else:
            # Restore the original source location if the optimization fails.
            self.location = original_location
    
        return self