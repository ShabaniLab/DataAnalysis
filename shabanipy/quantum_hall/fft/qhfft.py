import numpy as np
import scipy.constants as cs

class ZeroCrossingError(Exception):
    pass

def convert_to_inverse_field(field,rxx,field_cutoffs=None,N=None,
                             plot_resampled=False):
    """Convert field from linearly spaced points in Tesla to linearly spaced
       points in 1/Tesla for use in FFT

    The conversion is done by linearly resampling rxx over linearly spaced
    points on the range 1/B as defined by the parameters provided. Resampling
    is done by via numpy.interp to lineraly obtain the new values.

    Parameters
    ----------
    field : np.ndarray
        Magnetic field values for which the the longitudinal resistance was
        measured. The values should all be either field > 0 or field < 0.
        This condition should either be applied before passing to this function
        or applied via the keyword argument 'field_cutoffs'.
        This can be a multidimensional array in which case the last
        dimension will be considered as the swept dimension.
    rxx : np.ndarray
        Longitudinal resistance values which were measured.
        This can be a multidimensional array in which case the last dimension
        will be considered as the swept dimension.
    field_cutoffs = None : tuple | np.ndarray
        Pair or pairs of (low,high) field values on which to exclude from the
        resampling. This can be used to adjust the range of points over which
        the resampling is done. If only one pair of values is provided it will
        be used for all data points.
        If the default value of 'None' is provided then the resampling will be
        done over the full range given.
    N : int
        The number of points over which to sample.  If the default value of
        'None' is provided then the resampling will be done for the number of
        field points provided with respect to limitations by field_cutoffs.

    
    Returns
    -------
    resampled_inverse_field : float | np.ndarray
        The linearly sampled 1/B points used to obtain resampled_rxx.
        
    resampled_rxx : float | np.ndarray
        The longitundinal resistance points linearly interpolated from rxx
        corresponding to the points in inverse_field 
    """
    if field_cutoffs is None:
       field_cutoffs = (None,None) 
    if len(field.shape) >= 2:
        original_shape = field.shape[:-1]
        trace_number = np.prod(original_shape)
        field = field.reshape((trace_number, -1))
        rxx = rxx.reshape((trace_number, -1))
        if len(field_cutoffs) == 2:
            fc = np.empty(original_shape + (2,))
            fc[..., 0] = field_cutoffs[0]
            fc[..., 1] = field_cutoffs[1]
            field_cutoffs = fc
        field_cutoffs = field_cutoffs.reshape((trace_number, -1))
    else:
        trace_number = 1
        field = np.array((field,))
        rxx = np.array((rxx,))
        field_cutoffs = np.array((field_cutoffs,))
    
    if N is None:
        points = field[0].size
    else:
        points = N

    resampled_inverse_field = np.empty((trace_number,points))
    resampled_rxx           = np.empty((trace_number,points))
    for i in range(trace_number):
        start_field, stop_field = field_cutoffs[i]
        if start_field is None or np.isnan(start_field):
            start_field = np.min(field)
        if stop_field is None or np.isnan(stop_field):
            stop_field = np.max(field)
        start_ind = np.argmin(np.abs(field[i] - start_field))
        stop_ind = np.argmin(np.abs(field[i] - stop_field))
        start_ind, stop_ind =\
            min(start_ind, stop_ind), max(start_ind, stop_ind)
        f = field[i][start_ind:stop_ind]
        r = rxx[i][start_ind:stop_ind]
        if np.all(f > 0) or np.all(f < 0):
            pass
        else:
            raise ZeroCrossingError("Field points should be all > 0 or < 0")
        inv_f  = 1/f
        rs_inv_f = np.linspace(np.min(inv_f),np.max(inv_f),points)
        if inv_f[0] < inv_f[-1]:
            _rxx = np.interp(rs_inv_f,inv_f,r)
        if inv_f[0] > inv_f[-1]:
            _rxx = np.interp(rs_inv_f,inv_f[::-1],r[::-1])
        
        if plot_resampled:
            plt.figure()
            plt.plot(inv_f, r, '+',label='data')
            plt.plot(rs_inv_f, _rxx,'.',label='sampled') 
            plt.xlabel('Field')
            plt.ylabel('Rxy')
            plt.tight_layout()

        resampled_inverse_field[i] = rs_inv_f
        resampled_rxx[i]           = _rxx
    return resampled_inverse_field,resampled_rxx

def sdh_fft(field,rxx,field_cutoffs=None,N=None,plot_fft=False):
    """Takes the FFT of SdH oscillations
    
    FFT is done with respect to inverse field.  Field and rxx are resampled
    using convert_to_inverse_field.

    Parameters
    ----------
    field : np.ndarray
        Magnetic field values for which the the longitudinal resistance was
        measured. The values should all be either field > 0 or field < 0.
        This condition should either be applied before passing to this function
        or applied via the keyword argument 'field_cutoffs'.
        This can be a multidimensional array in which case the last
        dimension will be considered as the swept dimension.
    rxx : np.ndarray
        Longitudinal resistance values which were measured.
        This can be a multidimensional array in which case the last dimension
        will be considered as the swept dimension.
    field_cutoffs (= None) : tuple | np.ndarray
        Pair or pairs of [low,high) field values to include in the FFT. This
        can be used to adjust the range of points over which the resampling is
        done. If only one pair of values is provided it will be used for all
        data sets.
        If the default value of 'None' is provided then the FFT will be
        done over the full range of provided data.
    N (= None) : int
        The number of points to have in the resampled data.  If the default
        value of 'None' is provided then the resampling will be done for the
        number of field points provided with respect to limitations by
        field_cutoffs.
    
    Returns
    -------
    frequency : float | np.ndarray
        The frequency bins returned in units of carrier density (cm^-2).  Array
        is shifted to have sorted order of increasing value from negative to
        positive.
        
    power_spectrum : float | np.ndarray
        The power spectrum obtained from the FFT. Array is shifted to respect
        order of frequency array.
    """
    inv_field,resampled_rxx = convert_to_inverse_field(field,rxx,
                                                       field_cutoffs,N)
    power_spectrum = np.empty(resampled_rxx.shape)
    frequency      = np.empty(resampled_rxx.shape)
    for i in range(resampled_rxx.shape[0]):
        fft  = np.fft.fft(resampled_rxx[i])
        fft  = np.fft.fftshift(fft)
        power_spectrum[i] = np.abs(fft)**2
    
        field_spacing = np.abs(inv_field[i][1]-inv_field[i][0])
        
        Bfreq = np.fft.fftfreq(inv_field[i].size,d=field_spacing)
        nfreq = cs.e*Bfreq/(np.pi*cs.hbar)/1e4

        frequency[i] = np.fft.fftshift(nfreq)
        if plot_fft:
            plt.figure()
            plt.plot(frequency[i], fft, '-',label='data')
            plt.xlabel('n')
            plt.ylabel('FFT')
            plt.tight_layout()

    return frequency,power_spectrum


