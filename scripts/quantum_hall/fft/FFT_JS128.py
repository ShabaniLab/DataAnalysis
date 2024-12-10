from shabanipy.utils.labber_io import LabberData
import matplotlib.pyplot as plt
from plot_fft import plot_fft
import numpy as np

B_FIELD_INDEX    = 0
V_GATE_INDEX     = 1
V_L3_REAL_INDEX  = 2
V_L3_IMAG_INDEX  = 3
V_L2_REAL_INDEX  = 4
V_L2_IMAG_INDEX  = 5
I_LEAKAGE_INDEX  = 6
T_MEASURED_INDEX = 7
V_L4_REAL_INDEX  = 8
V_L4_IMAG_INDEX  = 9

fname = '/Users/javadshabani/Desktop/Projects/data_for_fft/JS128HB_MD001_007.hdf5'

with LabberData(fname) as data:
    shape = data.compute_shape([B_FIELD_INDEX,V_GATE_INDEX])
    sets  = [data.get_data(i) for i in range(10)]
    B_field,V_gate  = sets[:2]
    V3_real,V3_imag = sets[2:4]
    V2_real,V2_imag = sets[4:6]
    T_measured      = sets[6]
    V4_real,V4_imag = sets[7:9]

B   = B_field.reshape(shape).T
Vg  = V_gate.reshape(shape).T
V3r = V3_real.reshape(shape).T
V3i = V3_imag.reshape(shape).T
V2r = V2_real.reshape(shape).T
V2i = V2_imag.reshape(shape).T
V4r = V4_real.reshape(shape).T
V4i = V4_imag.reshape(shape).T
Tm  = T_measured.reshape(shape).T

I = 50e-9

Rxy   = V3r/I
Rxy_i = V3i/I
Rxx   = V2r/I
Rxx_i = V2i/I
Ryy   = V4r/I
Ryy_i = V4i/I

xlims = (1e10,5e12)
zlims = (0,5e9)
nlims = (0,4)

sample = 'JS128'

plot_fft(B,Vg,Rxx,Rxy,xlims,zlims,nlims,sample,
         Blims=(2.5,None),
         plot_traces=False,
         plot_fft_traces=False,
         plot_fft_with_density=True,
         plot_fft_with_density_math=True)
plt.show()
