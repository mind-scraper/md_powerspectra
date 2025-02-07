from md_powerspectra import md_powerspectra

traj_name = "traj.dump"

powerspectra = md_powerspectra(
                                traj_name = traj_name,
                                format = "lammps",                                 
                                specorder = ["C", "O"], #To correctly consider mass from lammps dump                                
                                indices=[0,1,2], # consider the first 3 atoms
                                dt=5, #timestep in fs, this is equal to your MD timestep * how often you save the trajectory
                                time=10, #in ps, this ditermine where to start calculating the spectra. The considered structure will be +/- Ntraj/2 of snapsot at time
                                Ntraj=1000, # consider 1000 trajectory
                                ref_vib = [622.5, 1300.7, 2329.6], #in cm^-1, from experiment or harmonic approx, this will be ploted just as reference
                                xlim = 5000, #plot until 5000 cm^-1,
                               )
powerspectra.run()   