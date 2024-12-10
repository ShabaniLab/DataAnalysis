from shabanipy.utils.labber_io import LabberData
from shabanipy.quantum_hall.fft import qhfft
from shabanipy.quantum_hall.density import extract_density
import matplotlib.pyplot as plt
import numpy as np


def plot_fft(B,Vg,Rxx,Rxy,xlims,zlims,n_lims,sample,
             Blims=None,
             plot_traces=False,
             plot_fft=False,
             plot_density_fits=False,
             plot_fft_traces = False,
             plot_fft_with_density = True,
             plot_fft_with_density_math = True):

    if Blims is None:
        Blims = (None,None)

    n,_ = extract_density(B,Rxy,n_lims,plot_fit=plot_density_fits)/1e4
    Vg  = [v[0] for v in Vg]
    
    freq,power = qhfft.sdh_fft(B,Rxx,field_cutoffs=Blims,plot_fft=plot_fft)

    if plot_traces:
        if Blims[0] is None:
            b0 = -np.inf
        else:
            b0 = Blims[0]

        if Blims[1] is None:
            b1 = np.inf
        else:
            b1 = Blims[1]
        
        plt.figure()
        for b,r in zip(B,Rxx):
            mask = np.logical_and(b0 <= b,b <= b1)
            plt.plot(b[mask],r[mask])
        plt.figure()
        for b,r in zip(B,Rxy):
            plt.plot(b,r)

    if len(freq.shape) >= 2 and freq.shape[0] > 1:
        fig,ax1 = plt.subplots(1,1)
        extent = (np.min(freq),np.max(freq),np.min(Vg),np.max(Vg))
        vmin,vmax = zlims
        im = ax1.imshow(power,extent=extent,vmin=vmin, vmax=vmax)
        ax1.set_xlim(xlims)
        extent = ax1.get_xlim() + ax1.get_ylim()
        ar = np.abs(extent[1]-extent[0])/np.abs(extent[3]-extent[2])
        ax1.set_aspect(ar)
        ax1.set_title(sample+' FFT')
        ax1.set_ylabel('Gate Voltage (V)')
        ax1.set_xlabel('Density (cm$^{-2}$)')
        plt.tight_layout()
        plt.savefig('/Users/javadshabani/Desktop/Projects/data_for_fft/'+
                    sample+'_fft.png')
       
        if plot_fft_with_density:
            fig,ax1 = plt.subplots(1,1)
            extent = (np.min(freq),np.max(freq),np.min(Vg),np.max(Vg))
            im = ax1.imshow(power,extent=extent,vmin=vmin, vmax=vmax)
            lims = (xlims[0],max(xlims[1],max(n)*1.1))
            ax1.set_xlim(lims)
            extent = ax1.get_xlim() + ax1.get_ylim()
            ar = np.abs(extent[1]-extent[0])/np.abs(extent[3]-extent[2])
            ax1.set_aspect(ar)
            ax1.set_title(sample+' FFT')
            ax1.set_ylabel('Gate Voltage (V)')
            ax1.set_xlabel('Density (cm$^{-2}$)')
            ax1.plot(n,Vg,'ro',label='n$_{Hall}$')
            ax1.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig('/Users/javadshabani/Desktop/Projects/data_for_fft/'+
                        sample+'_fft_with_hall_n.png')
        
        if plot_fft_with_density_math:
            fig,ax1 = plt.subplots(1,1)
            extent = (np.min(freq),np.max(freq),np.min(Vg),np.max(Vg))
            im = ax1.imshow(power,extent=extent,vmin=vmin, vmax=vmax)
            lims = (xlims[0],max(xlims[1],max(n)*1.1))
            ax1.set_xlim(lims)
            extent = ax1.get_xlim() + ax1.get_ylim()
            ar = np.abs(extent[1]-extent[0])/np.abs(extent[3]-extent[2])
            ax1.set_aspect(ar)
            ax1.set_title(sample+' FFT')
            ax1.set_ylabel('Gate Voltage (V)')
            ax1.set_xlabel('Density (cm$^{-2}$)')
            n_fft = []
            for f,p in zip(freq,power):
                mask = np.logical_and(lims[0] < f, f < lims[1])
                n_fft.append(f[mask][np.argmax(p[mask])])
            ax1.plot(n,Vg,'ro',label='n$_{Hall}$')
            ax1.plot(n_fft,Vg,'bo',label='n$_{FFT}$')
            ax1.plot(n-n_fft,Vg,'go',label='n$_{Hall}$ - n$_{FFT}$')
            ax1.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig('/Users/javadshabani/Desktop/Projects/data_for_fft/'+
                        sample+'_fft_with_hall_n_math.png')
       
        if plot_fft_traces:
            fig,ax1 = plt.subplots(1,1)
            offset = 0
            for f,p,v in zip(freq[::-1],power[::-1],Vg[::-1]):
                mask = np.logical_and(f > lims[0], f < lims[1])
                f = f[mask]
                p = p[mask]
                ax1.plot(f,offset + p/max(p))
                ax1.text(f[-1],offset,f'Vg = {v}')
                offset += 5
            ax1.set_xlim()
