""" 
###MD Power Spectra###
This class calculate vibrational frequency from molecular dynamics trajectory using power spectrum.
In general, the steps are as follows
1st: extracting velocity (and masses)
2nd: calculate velocity autocorrelation function
3rd: perform fourier transform for the autocorrelation function

Reference: Martin Thomas et al. "Computing vibrational spectra from ab initio molecular dynamics", Phys. Chem. Chem. Phys., 2013,15, 6608-6622 (https://doi.org/10.1039/C3CP44302G)

Written by
Samuel Eka Putra Payong Masan
Ph.D. student at Morikawa Group
Osaka University
February 2025
######

Input parameter

    traj_name: str
        default=None
        Trajectory name.

    format: str
        default=None
        options: "lammps", None
        The trajectory format. For lammps, please specify the specorder to correctly account the atomic mass.         
    
        if format == "lammps":
        
            specorder: list of str
                default=[None]
                List of atomic species in lammps dump file.  

    indices: list of int
        default=[None]
        Index of atoms to be considered. 

    dt: float #in fs
        default=None
        Timestep of the trajectory. 
        This is equal to the timestep in molecular dynamics symultaion times how often the trajectory is saved.

    Ntraj: int
        default=1000
        Text     

    time: float #in ps
        default=None
        A fraction of total time to calculate the frequency. 
        For example, if your trajectory contain 100 ps simulation, by setting time=1, you will get frequency at around 1 ps.             

    ref_vib: list of float #in cm^-1
        default=[None]
        Reference frequency from experiment or harmonic approximation. 
        This will only be ploted as reference and will not affect the calculation.

    xlim: float #in cm^-1
        default=None
        Limit of the frequency plot.

Example        
    >>> from md_powerspectra import md_powerspectra

    >>> traj_name = "../traj.dump"
    >>> powerspectra = md_powerspectra(
                                traj_name = traj_name,
                                indices=[0,1,2], # consider the first 3 atoms
                                dt=5, #timestep in fs, this is equal to your MD timestep * how often you save the trajectory
                                time=10, #in ps, this ditermine where to start calculating the spectra. The considered structure will be +/- Ntraj/2 of snapsot at time
                                Ntraj=1000, # consider 1000 trajectory
                                ref_vib = [622.5, 1300.7, 2329.6], #in cm^-1, from experiment or harmonic approx, this will be ploted just as reference
                                xlim = 5000, #plot until 5000 cm^-1,
                                format = "lammps", 
                                specorder = ["C", "O"], #To correctly consider mass from lammps dump
                               )
    >>> powerspectra.run()        
"""

import numpy as np
from ase import units
from ase.io import read
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

class md_powerspectra():
    def __init__(self, 
                 traj_name = None, 
                 indices=[None], 
                 dt=None, 
                 time=None, 
                 Ntraj=1000, 
                 ref_vib = [None],
                 xlim = 5000,
                 format = None,
                 specorder = [None]
                 ):
        
        #Declaring variables
        self.dt = dt
        self.time = time
        self.Ntraj = int(Ntraj)
        self.ref_vib = ref_vib
        self.xlim = xlim

        #Cut the trajectory
        final = self.time * 1000 / self.dt
        final += self.Ntraj/2
        initial = final - self.Ntraj
        initial = int(initial)
        final = int(final)
        if initial < 0:
            raise ValueError("Please provide more trajectory ir reduce the time!")
        else:
            print(f"Using trajectory {initial} to {final}.")
        
        if format == "lammps":
            if specorder[0]:
                self.traj = read(traj_name, format='lammps-dump-text', specorder=specorder, index=slice(initial,final))
            else:
                raise ValueError("Please provide specorder for lammps trajectory!")
        else:
            self.traj = read(traj_name, index=slice(initial,final))

        if indices[0]:
            for atoms in self.traj:
                del atoms[indices]

        self.n_frames = len(self.traj)
        self.natoms = len(self.traj[0])

    def get_mass_weighted_velocities(self):                
        print("Calculating mass weighted velocities . . .")
        #Get atomic masses
        masses = np.array(self.traj[0].get_masses()).reshape(1, self.natoms, 1)  # Shape: (1, n_atoms, 1)

        #Get velocities
        velocities = np.array([atoms.get_velocities() for atoms in self.traj])  # Shape: (n_frames, n_atoms, 3)

        # Apply mass-weighting to velocities
        mass_weighted_velocities = velocities * np.sqrt(masses)  # Mass-weighted velocities

        # Flatten velocities for VACF computation
        return mass_weighted_velocities.reshape(self.n_frames, -1)  # Shape: (n_frames, n_atoms*3)
    
    def get_vacf(self):        
        mass_weighted_velocities = self.get_mass_weighted_velocities()
        print("Calculating velocity autocorrelation function . . .")
        vacf = np.zeros(self.n_frames)
        for t in range(self.n_frames):
            vacf[t] = np.sum([
                            np.dot(mass_weighted_velocities[i], mass_weighted_velocities[i + t]) 
                            for i in range(self.n_frames - t)
                            ])
        #Smoothen
        window = np.hanning(len(vacf))
        vacf *= window

        return vacf
    
    def do_fft(self):        
        frequencies = np.fft.rfftfreq(self.n_frames, d=self.dt*units.fs/units.s)
        frequencies = frequencies / (units._c * 100)  
        vacf = self.get_vacf()
        print("Performing fourier transform . . .")
        spectra = np.abs(np.fft.rfft(vacf))  # Compute Fourier Transform magnitude
        spectra = spectra/sum(spectra)
        # Save results
        print(f"Saving the results to power_spectra-{self.time}ps.log")
        np.savetxt(f"power_spectra-{self.time}ps.log", np.column_stack((frequencies, spectra)),
                   header="Frequency(cm^-1) Spectra")   

        return frequencies, spectra
    
    def plot_result(self):        
        if self.ref_vib[0]:
            for vib in self.ref_vib:
                plt.axvline(vib, linestyle='--', alpha=0.5, color='darkblue')

        frequencies, spectra = self.do_fft()
        print(f"Ploting the results to power_spectra-{self.time}ps.png")
        plt.plot(frequencies, spectra, color='darkred', alpha=0.75)
        plt.xlabel("Frequency (cm$^{-1}$)")
        plt.ylabel("Intensity")
        plt.legend()
        if max(frequencies) > self.xlim:
            plt.xlim(self.xlim*(-0.01),self.xlim)
        plt.title("Vibrational Spectra")
        plt.savefig(f'power_spectra-{self.time}ps.png', dpi=300, bbox_inches='tight', transparent=False)
    
    def run(self):
        self.plot_result()