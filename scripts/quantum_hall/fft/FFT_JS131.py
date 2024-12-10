from shabanipy.utils.labber_io import LabberData
from plot_fft import plot_fft
import matplotlib.pyplot as plt
import numpy as np

B_FIELD_INDEX    = 0
V_GATE_INDEX     = 1

fname = '/Users/javadshabani/Desktop/Projects/data_for_fft/JS131HB-MD002-004.hdf5'

with LabberData(fname) as data:
    shape = data.compute_shape([B_FIELD_INDEX,V_GATE_INDEX])
    sets  = [data.get_data(i) for i in range(10)]
    B_field,V_gate  = sets[:2]
    V4_real,V4_imag = sets[2:4]
    V3_real,V3_imag = sets[4:6]
    V1_real,V1_imag = sets[6:8]
    I_leakage       = sets[8]
    T_measured      = sets[9]

B   = B_field.reshape(shape).T
Vg  = V_gate.reshape(shape).T
V4r = V4_real.reshape(shape).T
V4i = V4_imag.reshape(shape).T
V3r = V3_real.reshape(shape).T
V3i = V3_imag.reshape(shape).T
V1r = V1_real.reshape(shape).T
V1i = V1_imag.reshape(shape).T
Tm  = T_measured.reshape(shape).T

I = 50e-9

Rxx   = V4r/I
Rxy   = V1r/I

xlims = (5e11,5e12)
zlims = (0,1e10)
nlims = (0,3)
Blims = (2,None)

sample = 'JS131'

plot_fft(B,Vg,Rxx,Rxy,xlims,zlims,nlims,sample,
         Blims=Blims,
         plot_traces=False,
         plot_fft_traces=False,
         plot_density_fits=False,
         plot_fft_with_density=True,
         plot_fft_with_density_math=True)
plt.show()
