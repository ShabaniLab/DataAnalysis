from shabanipy.utils.labber_io import LabberData
import matplotlib.pyplot as plt
from plot_fft import plot_fft
import numpy as np

B_FIELD_INDEX    = 0
V_GATE_INDEX     = 1
T_TARGET_INDEX   = 2
V_L2_REAL_INDEX  = 3
V_L2_IMAG_INDEX  = 4
V_L1_REAL_INDEX  = 5
V_L1_IMAG_INDEX  = 6
T_MEASURED_INDEX = 7

fname = '/Users/javadshabani/Desktop/Projects/data_for_fft/JS138_124HB_BM003_007.hdf5'
temps = 5

with LabberData(fname) as data:
    shape    = data.compute_shape([B_FIELD_INDEX,V_GATE_INDEX,T_TARGET_INDEX])
    shape[2] = temps
    sets     = [data.get_data(i) for i in range(8)]
    
    B_field,V_gate  = sets[:2]
    T_target        = sets[2]
    V2_real,V2_imag = sets[3:5]
    V1_real,V1_imag = sets[5:7]
    T_measured      = sets[7]

N = len(B_field)//shape[0]

d1 = (shape[0],N)
d2 = [shape[2],shape[1],shape[0]]
B   = B_field.reshape(d1)[:,:shape[1]*shape[2]].T.reshape(d2)
Vg  = V_gate.reshape(d1)[:,:shape[1]*shape[2]].T.reshape(d2)
Tt  = T_target.reshape(d1)[:,:shape[1]*shape[2]].T.reshape(d2)
V2r = V2_real.reshape(d1)[:,:shape[1]*shape[2]].T.reshape(d2)
V2i = V2_imag.reshape(d1)[:,:shape[1]*shape[2]].T.reshape(d2)
V1r = V1_real.reshape(d1)[:,:shape[1]*shape[2]].T.reshape(d2)
V1i = V1_imag.reshape(d1)[:,:shape[1]*shape[2]].T.reshape(d2)
Tm  = T_measured.reshape(d1)[:,:shape[1]*shape[2]].T.reshape(d2)

I = 1e-6

Rxx   = V2r/I
Rxx_i = V2i/I
Rxy   = V1r/I
Rxy_i = V1i/I

# use resistivity instead of resistance
# Rxx = rhoxx

# flip and arrange data
# trim data which is unneccessary

Rxx = Rxx[0][::-1]
Rxy = Rxy[0][::-1]
B   = B[0][::-1]
Vg  = Vg[0][::-1]

xlims = (3e11,5e12)
zlims = (0,5e7)
nlims = (0,3)

sample = 'JS124'

plot_fft(B,Vg,Rxx,Rxy,xlims,zlims,nlims,sample,
         Blims=(.8,None),
         plot_traces=False,
         plot_fft_traces=False,
         plot_fft_with_density=True,
         plot_fft_with_density_math=True)
plt.show()
    

