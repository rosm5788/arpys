import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from scipy.optimize import curve_fit
from typing import List,Optional
import h5py

def colored_line(x, y, c, ax, **lc_kwargs): 
    # Stole this from https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)
    vmin = default_kwargs.pop('vmin',None)
    vmax = default_kwargs.pop('vmax',None)
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)
    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment
    lc.set_clim(vmin=vmin,vmax=vmax)
    return ax.add_collection(lc)

class Parser():
    initialzed = False
    def __init__(self,directory,Ef=None):
        self.directory = directory
        self.e_fermi = Ef
        self.subtracted = False
        self.parse()
        self.k_reciprocal_to_cartesian()
        self.initialzed = True
        
    def parse(self): # Override this with something to actually do the parsing in a subclass for a specific parser
        raise NotImplementedError("This parse method should be overridden by something for a specific file type")
        # Populate self.nkpoints, self.nbands, self.e_fermi as numbers
        # Populate self.k_points_reciprocal as (self.nkpoints,3) size np.array
        # Populate self.band_energies, self.band_occupations as (self.nkpoints,self.nbands) size np.array
        # Populate self.spins as (4,self.nkpoints,self.nbands) size np.array. First element should be (spin_magnitude,sx,sy,sz)
        # Populate self.atom_types and self.lm_labels as a list of strings and self.atom_num as int
        # Populate self.atom_projections as dict['atom_type'] with (num_orbitals,self.nkpoints,self.nbands) size np.array for each atom type in the dictionary
        # If you don't populate any of those, the plotter will handle it fine as long as you don't request plotting features that require them

    def get_bands_xarray(self,cartesian = True):
        if self.subtracted == False: 
            print("Warning: The band energies haven't been subtracted by Ef")
        if cartesian: kpoints = self.kpoints_cartesian
        else: kpoints = self.kpoints_reciprocal
        kx_points = np.unique(kpoints[:,0])
        ky_points = np.unique(kpoints[:,1])
        kz_points = np.unique(kpoints[:,2])
        bands = xr.DataArray(np.zeros((len(kx_points),len(ky_points),len(kz_points),self.nbands)),coords=[kx_points,ky_points,kz_points,range(1,self.nbands+1)],dims=["kx","ky","kz","band"])
        for i in range(self.nkpoints):
            bands.loc[kpoints[i,0],kpoints[i,1],kpoints[i,2]] = self.band_energies[i]
        return bands

    def get_occupation_xarray(self,cartesian = True):
        if cartesian: kpoints = self.kpoints_cartesian
        else: kpoints = self.kpoints_reciprocal
        kx_points = np.unique(kpoints[:,0])
        ky_points = np.unique(kpoints[:,1])
        kz_points = np.unique(kpoints[:,2])
        occupations = xr.DataArray(np.zeros((len(kx_points),len(ky_points),len(kz_points),self.nbands)),coords=[kx_points,ky_points,kz_points,range(1,self.nbands+1)],dims=["kx","ky","kz","band"])
        for i in range(self.nkpoints):
            occupations.loc[kpoints[i,0],kpoints[i,1],kpoints[i,2]] = self.band_occupations[i]
        return occupations
    
    def get_spin_xarray(self,component,cartesian = True):
        # This function lets you refer to the spin components as 1,2,3 or 'x','y','z'
        if cartesian: kpoints = self.kpoints_cartesian
        else: kpoints = self.kpoints_reciprocal
        kx_points = np.unique(kpoints[:,0])
        ky_points = np.unique(kpoints[:,1])
        kz_points = np.unique(kpoints[:,2])
        if type(component) is str:
            if component == 'x':
                component = 1
            elif component == 'y':
                component = 2
            elif component == 'z':
                component = 3
            else:
                raise KeyError("Valid Spins are: 1,2,3 or 'x','y','z'")
        elif component > 3 or component < 1 or component % 1 != 0:
            raise KeyError("Valid Spins are: 1,2,3 or 'x','y','z'")
        component = int(component)
    
        spin = xr.DataArray(np.zeros((len(kx_points),len(ky_points),len(kz_points),self.nbands)),coords=[kx_points,ky_points,kz_points,range(1,self.nbands+1)],dims=["kx","ky","kz","band"])
        for i in range(self.nkpoints):
            spin.loc[kpoints[i,0],kpoints[i,1],kpoints[i,2]] = self.spins[component,i]/self.spins[0,i]
        return spin
    
    def get_atom_projections_xarray(self,cartesian = True):
        if cartesian: kpoints = self.kpoints_cartesian
        else: kpoints = self.kpoints_reciprocal
        kx_points = np.unique(kpoints[:,0])
        ky_points = np.unique(kpoints[:,1])
        kz_points = np.unique(kpoints[:,2])
        if not hasattr(self, 'atom_projections'):
            raise NotImplementedError("This parser does not extract atom projections.")
        projections = {}
        for atom in self.atom_types:
            projections[atom] = xr.DataArray(np.zeros((len(kx_points),len(ky_points),len(kz_points),self.nbands,len(self.atom_projections[atom]))),coords=[kx_points,ky_points,kz_points,range(1,self.nbands+1),self.lm_labels[0:len(self.atom_projections[atom])]],dims=["kx","ky","kz","band","orbital"])
            for i in range(self.nkpoints):
                projections[atom].loc[kpoints[i,0],kpoints[i,1],kpoints[i,2]] = self.atom_projections[atom][:,i].T/self.spins[0,i,np.newaxis].T
        return projections
    
    def get_site_projections_xarray(self,cartesian = True):
        if cartesian: kpoints = self.kpoints_cartesian
        else: kpoints = self.kpoints_reciprocal
        kx_points = np.unique(kpoints[:,0])
        ky_points = np.unique(kpoints[:,1])
        kz_points = np.unique(kpoints[:,2])
        if not hasattr(self, 'sites'):
            raise NotImplementedError("This parser does not extract site projections")
        projections = xr.DataArray(np.zeros((len(kx_points),len(ky_points),len(kz_points),self.nbands,len(self.sites))),coords=[kx_points,ky_points,kz_points,range(1,self.nbands+1),range(len(self.sites))],dims=["kx","ky","kz","band","site"])
        for i in range(self.nkpoints):
            projections.loc[kpoints[i,0],kpoints[i,1],kpoints[i,2]] = self.sites[:,i].T/self.spins[0,i,np.newaxis].T
        return projections

class EIGENVAL_Parser(Parser):
    NKPOINTS_LINE = 5
    FIRST_K_LINE = 7
    initialized = False
    dos = None
    
    def __init__(self,directory,Ef=None):
        try:
            from pyprocar.io.vasp import Outcar
            self.Outcar = Outcar
        except ImportError:
            raise ImportError("You need to have pyprocar installed to use the EIGENVAL Parser")
        self.directory = directory
        self.e_fermi = Ef
        self.subtracted = False
        self.parse()
        self.k_reciprocal_to_cartesian()
        self.initalized = True

    def parse(self):
        if not os.path.exists(self.directory+os.sep+"EIGENVAL"): 
            raise OSError("Can't find the EIGENVAL file in that directory")
        with open(self.directory+os.sep+"EIGENVAL", "rt") as file:
            text = file.readlines()
        self.nkpoints, self.nbands = [int(elem) for elem in text[self.NKPOINTS_LINE].split()[1:3]]
        self.kpoints_reciprocal = np.zeros((self.nkpoints,3))
        self.kpoints_cartesian = np.zeros((self.nkpoints,3))
        self.band_energies = np.zeros((self.nkpoints,self.nbands))
        self.band_occupations = np.zeros((self.nkpoints,self.nbands))
        print(f"EIGENVAL file found with {self.nkpoints} k-points and {self.nbands} bands")
        k_point_counter = 0
        for i in range(self.FIRST_K_LINE,len(text)):
            index = (i - self.FIRST_K_LINE) % (self.nbands + 2)
            if index == 0:
                self.kpoints_reciprocal[k_point_counter] = np.array([float(kdim) for kdim in text[i].split()[0:3]])
            elif index == self.nbands + 1:
                k_point_counter += 1
            else:
                self.band_energies[k_point_counter,index-1], self.band_occupations[k_point_counter,index-1] = [float(elem) for elem in text[i].split()[1:3]]
                

    def subtract_ef(self):
        if self.subtracted == True:
            print("The energies have already been subtracted by the Fermi Energy and I'm not doing it again")
            return False
        if self.e_fermi is None:
            if not os.path.exists(self.directory+os.sep+"OUTCAR"): 
                raise OSError("Can't find an OUTCAR file to get Fermi Energy in EIGENVAL directory (you can specify Ef when initializing the class)")
            self.e_fermi = self.Outcar(self.directory+os.sep+"OUTCAR").efermi
        print(f"Fermi Energy: {self.e_fermi:.3f} eV")
        self.band_energies -= self.e_fermi
        if hasattr(self,'dos_energies') and self.dos_energies is not None:
            self.dos_energies -= self.e_fermi
        self.subtracted = True
        return self.e_fermi

    def k_reciprocal_to_cartesian(self,rotation=None): # By default, the file gives k points in reciprocal lattice so I have this run by default to fix that
        OUTCAR_path = self.directory + os.sep+"OUTCAR"
        if not os.path.exists(OUTCAR_path): 
            raise OSError("Can't find the OUTCAR file in the directory to get reciprocal lattice")
        reciprocal_vectors = self.Outcar(OUTCAR_path).reciprocal_lattice
        self.reciprocal_lattice = reciprocal_vectors
        print("Reciprocal Lattice Vectors:\n",reciprocal_vectors)
        for i in range(self.nkpoints):
            self.kpoints_cartesian[i] = np.round(sum([self.kpoints_reciprocal[i,j]*reciprocal_vectors[j]*2*np.pi for j in range(3)]),6)
        if rotation is not None:
            theta = np.arccos(rotation[2]/np.linalg.norm(rotation))
            phi = np.sign(rotation[1])*np.arccos(rotation[0]/np.sqrt(rotation[0]**2+rotation[1]**2))
            inv_rot_matrix = np.linalg.inv(np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]]) @ np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]))
            for i in range(len(self.kpoints_cartesian)):
                self.kpoints_cartesian[i] = np.round(inv_rot_matrix @ self.kpoints_cartesian[i],6)
    
    def get_spin_xarray(self, *args, **kwargs):
        raise NotImplementedError("EIGENVAL_Parser can't get spins")
    
    def get_atom_projections_xarray(self, *args, **kwargs):
        raise NotImplementedError("EIGENVAL_Parser can't get atom projections")
    
    def parse_DOSCAR(self):
        if not os.path.exists(self.directory+os.sep+"DOSCAR"): 
            raise OSError("Can't find the DOSCAR file in given directory")
        with open(self.directory+os.sep+"DOSCAR", "rt") as file:
            text = file.readlines()
        points = int(text[5].split()[2])
        dos = np.zeros((points,2))
        for i in range(points):
            dos[i] = [float(elem) for elem in text[i+6].split()[0:2]]
        self.dos = dos[:,0]
        self.dos_energies = dos[:,1]
    
    def plot_DOS(self,pltfigax=None,**plot_kwargs):
        if self.dos is None:
            self.parse_DOSCAR()
        if pltfigax is None:
            fig, ax = plt.subplots()
        if not self.subtracted:
            self.subtract_ef()
        else: fig, ax = pltfigax
        ax.plot(self.dos,self.dos_energies,**plot_kwargs)
        ax.set_xlabel("Binding Energy (eV)")
        ax.set_ylabel("Density of States")
        return fig, ax

class vaspouth5_Parser(Parser):
    initialzed = False
    def __init__(self,directory,Ef=None,rotation=None):
        self.directory = os.path.expanduser(directory)
        self.e_fermi = Ef
        self.subtracted = False
        self.parse()
        self.k_reciprocal_to_cartesian(rotation)
        self.rotation = rotation
        self.band_plot = self.parse_bandplotinfo()
        self.subtract_ef()
        self.initialzed = True
        
    def parse(self):
        if not os.path.exists(self.directory): 
            raise OSError("You gave me a fake directory")
        if not os.path.exists(self.directory+os.sep+"vaspout.h5"): 
            raise OSError("Can't find the vaspout.h5 file in that directory")
        self.file = h5py.File(self.directory+os.sep+"vaspout.h5")
        self.nkpoints = self.file['results']['electron_eigenvalues']['kpoints'][()]
        self.nbands = self.file['results']['electron_eigenvalues']['nb_tot'][()]
        self.kpoints_reciprocal = self.file['results']['electron_eigenvalues']['kpoint_coords'][:]
        self.kpoints_cartesian = np.zeros((self.nkpoints,3))
        if self.e_fermi is None:
            self.e_fermi = float(self.file['results']['electron_dos']['efermi'][()])
        self.lattice_vectors = self.file['input']['poscar']['lattice_vectors'][:]
        self.band_energies = self.file['results']['electron_eigenvalues']['eigenvalues'][0]
        self.band_occupations = self.file['results']['electron_eigenvalues']['fermiweights'][0]
        self.dos = self.file['results']['electron_dos']['dos'][0]
        self.dos_integrated = self.file['results']['electron_dos']['dosi'][0]
        self.dos_energies = self.file['results']['electron_dos']['energies'][:]
        self.spins = np.sum(self.file['results']['projectors']['par'][:],axis=(1,2))

        self.atom_types = [atom_type.decode().split('_')[0] for atom_type in self.file['input']['poscar']['ion_types'][:]]
        self.atom_num = self.file['input']['poscar']['number_ion_types'][:]
        orbitals = self.file['results']['projectors']['par'][0]
        projected_dos = self.file['results']['electron_dos']['dospar'][0]
        self.sites = np.sum(orbitals,axis=1)
        self.atom_projections = {}
        self.dos_projected = {}
        for i in range(len(self.atom_types)):
            if self.atom_types[i] in self.atom_projections: # This sees if it's already made a dict entry for an atom and if so, adds to that rather than making a new one
                self.atom_projections[self.atom_types[i]] += np.sum(orbitals[np.sum(self.atom_num[0:i]):np.sum(self.atom_num[0:i+1])],axis=0)
                self.dos_projected[self.atom_types[i]] += np.sum(projected_dos[np.sum(self.atom_num[0:i]):np.sum(self.atom_num[0:i+1])],axis=0)
            else:
                self.atom_projections[self.atom_types[i]] = np.sum(orbitals[np.sum(self.atom_num[0:i]):np.sum(self.atom_num[0:i+1])],axis=0)
                self.dos_projected[self.atom_types[i]] = np.sum(projected_dos[np.sum(self.atom_num[0:i]):np.sum(self.atom_num[0:i+1])],axis=0)

        self.lm_labels = [lm_label.decode().strip() for lm_label in self.file['results']['projectors']['lchar'][:]]
        
        material_name = ""
        for i in range(len(self.atom_types)):
            material_name = material_name+self.atom_types[i]+str(self.atom_num[i])
        print(f"vaspout.h5 file found for {material_name} with {self.nkpoints} k-points and {self.nbands} bands")
    
    def subtract_ef(self):
        if self.subtracted == True:
            print("The energies have already been subtracted by the Fermi Energy and I'm not doing it again")
            return False
        print(f"Fermi Energy: {self.e_fermi:.3f} eV")
        self.band_energies -= self.e_fermi
        self.dos_energies -= self.e_fermi
        self.subtracted = True
        return self.e_fermi
    
    def change_ef(self,new_ef:float):
        if not self.subtracted:
            self.subtract_ef()
        print(f"Changing Ef from {self.e_fermi:.4f} eV to {new_ef:.4f} eV")
        self.band_energies += self.e_fermi
        self.dos_energies += self.e_fermi
        self.e_fermi = new_ef
        self.band_energies -= self.e_fermi
        self.dos_energies -= self.e_fermi

    def parse_bandplotinfo(self):
        if self.file['input']['kpoints']['mode'][()].decode() != 'l': # If it's not in line mode, no reason to get band stuff
            return False
        kpoint_labels_raw = [label.decode().strip() for label in self.file['input']['kpoints']['labels_kpoints'][:]]
        kpoint_labels_raw = [s.replace("\\\\", "\\") for s in kpoint_labels_raw]
        self.kpoint_labels_raw = kpoint_labels_raw
        self.kpoint_labels = [kpoint_labels_raw[0]] # VASP repeats kpoints between lines but I only want each one labeled once
        for i in range(1,len(kpoint_labels_raw)-1,2):
            if kpoint_labels_raw[i] == kpoint_labels_raw[i+1]:
                self.kpoint_labels.append(kpoint_labels_raw[i])
            else: # This is for the case where the lines next to each other don't share a kpoint
                self.kpoint_labels.append(kpoint_labels_raw[i]+"/"+kpoint_labels_raw[i+1])
        self.kpoint_labels.append(kpoint_labels_raw[-1])

        kpoints_per_line = int(self.file['input']['kpoints']['number_kpoints'][()])
        self.kpoints_per_line = kpoints_per_line
        self.kpoint_indices = [i*kpoints_per_line for i in range(len(self.kpoint_labels))]
        self.kpoint_indices[-1] -= 1 # This is because the last k point isn't repeated

        self.nlines = len(kpoint_labels_raw)//2
        kpoint_labels_coords = np.array(self.file['input']['kpoints']['coordinates_kpoints'][:])
        kpoint_labels_coords_cart = np.zeros((len(kpoint_labels_coords),3))
        for i in range(len(kpoint_labels_coords)):
            kpoint_labels_coords_cart[i] = sum([kpoint_labels_coords[i,j]*self.reciprocal_lattice[j] for j in range(3)])
        self.line_lengths = [np.linalg.norm(kpoint_labels_coords_cart[2*i+1]-kpoint_labels_coords_cart[2*i]) for i in range(self.nlines)]
        return True
    
    def k_reciprocal_to_cartesian(self,rotation=None): # By default, the file gives k points in reciprocal lattice so I have this run by default to fix that
        a1 = self.lattice_vectors[0]
        a2 = self.lattice_vectors[1]
        a3 = self.lattice_vectors[2]
        V = np.dot(a1,np.cross(a2,a3))
        b1 = 2*np.pi/V * np.cross(a2,a3)
        b2 = 2*np.pi/V * np.cross(a3,a1)
        b3 = 2*np.pi/V * np.cross(a1,a2)
        reciprocal_vectors = np.array([b1,b2,b3])
        self.reciprocal_lattice = reciprocal_vectors
        print("Reciprocal Lattice Vectors:\n",reciprocal_vectors/(2*np.pi))
        for i in range(self.nkpoints):
            self.kpoints_cartesian[i] = np.round(sum([self.kpoints_reciprocal[i,j]*reciprocal_vectors[j] for j in range(3)]),6)
        if rotation is not None:
            theta = np.arccos(rotation[2]/np.linalg.norm(rotation))
            phi = np.sign(rotation[1])*np.arccos(rotation[0]/np.sqrt(rotation[0]**2+rotation[1]**2))
            inv_rot_matrix = np.linalg.inv(np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]]) @ np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]]))
            for i in range(len(self.kpoints_cartesian)):
                self.kpoints_cartesian[i] = np.round(inv_rot_matrix @ self.kpoints_cartesian[i],4)

    def plot_bands(self,color:str=None,bands_to_plot:list=None,k_range:tuple=None,E_range:tuple=None,scaled:bool=True,proj_atom:str=None,proj_orbital:str=None,spin=None,proj_sites:list=None,pltfigax=None,**plot_kwargs):
        """ Plots the full DFT band structure from a VASP run in line mode
        :param color: Color of the plotted bands. If you do orbital projection, you can put in either a color (which will be used in a transparent to opaque color map) or a matplotlib cmap
        :param bands_to_plot: Tuple, list, or range of the bands you want to plot
        :param k_range: Tuple of the limits of the index of high symmetry points you want to plot over 
        :param E_range: Tuple that sets ylim on the plot
        :param scaled: Whether or not you want the x axis scaled by the distance between the kpoints in cartesian reciprocal space
        :param proj_atom: Atom to do orbital projection onto
        :param proj_orbital: Orbital to project onto, can be s,p,d,f or specific orbital names. None or 'total' will do overall atom projection
        :param spin: Spin component to project onto, can be 'x','y','z' or 1,2,3
        :param proj_sites: List or tuple of sites you want to project onto
        :param pltfigax: Tuple of matplotlib fig and ax you want to plot on like (fig,ax)
        :param plot_kwargs: Excess parameters are passed to the DFT bands plot call (like vmin,vmax,cbar_title for the colorbar)
        """
        if not self.band_plot:
            raise NotImplementedError("This vaspout.h5 file wasn't made from a VASP run in line mode and can't be used to plot bands")
        if not self.subtracted:
            self.subtract_ef()
        if pltfigax is None:
            fig,ax = plt.subplots()
        else: fig, ax = pltfigax
        if bands_to_plot is None:
            bands_to_plot = range(self.nbands)
        elif not (isinstance(bands_to_plot,range) or isinstance(bands_to_plot,list)):
            raise ValueError("bands_to_plot should be either range or list")
        
        # Sets up the x axis so the kpoints are scaled by the cartesian distance in k-space between kpoints
        x_axis = []
        if scaled:
            for i in range(self.nlines):
                if i != self.nlines - 1:
                    x_axis.extend(np.linspace(sum(self.line_lengths[0:i]),sum(self.line_lengths[0:i+1]),self.kpoint_indices[i+1]-self.kpoint_indices[i],endpoint=False))
                else: x_axis.extend(np.linspace(sum(self.line_lengths[0:i]),sum(self.line_lengths[0:i+1]),self.kpoint_indices[i+1]-self.kpoint_indices[i]+1,endpoint=True))
        else:
            for i in range(self.nlines):
                if i != self.nlines - 1:
                    x_axis.extend(np.linspace(i,i+1,self.kpoint_indices[i+1]-self.kpoint_indices[i],endpoint=False))
                else: x_axis.extend(np.linspace(i,i+1,self.kpoint_indices[i+1]-self.kpoint_indices[i]+1,endpoint=True))
        
        # If we're going to project onto spin, site, or atom/orbital, this is the chunk where we get that data
        proj_data = None
        if proj_atom is not None:
            if proj_orbital == "s":
                proj_data = self.atom_projections[proj_atom][0]
            elif proj_orbital == "p":
                proj_data = np.sum(self.atom_projections[proj_atom][1:4],axis=0)
            elif proj_orbital == "d":
                proj_data = np.sum(self.atom_projections[proj_atom][4:9],axis=0)
            elif proj_orbital == "f":
                proj_data = np.sum(self.atom_projections[proj_atom][9:16],axis=0)
            elif proj_orbital == "total" or proj_orbital is None:
                proj_data = np.sum(self.atom_projections[proj_atom],axis=0)
            elif proj_orbital not in self.lm_labels:
                raise KeyError("That orbital name isn't in VASP")
            else:
                proj_data = self.atom_projections[proj_atom][self.lm_labels.index(proj_orbital)]

            proj_data /= self.spins[0]
            default_plot_kwargs = {'color':'afmhot_r','vmin':0,'vmax':1,'cbar_title':f"Projection onto {proj_atom} {proj_orbital if not (proj_orbital is None or proj_orbital=='total') else ''} orbital"}
        elif proj_orbital:
            raise ValueError("I can't plot an orbital unless you tell me what atom we're workign with")
            
        elif spin is not None:
            if spin in ['x','sx',1]:
                spin = 1
            elif spin in ['y','sy',2]:
                spin = 2
            elif spin in ['z','sz',3]:
                spin = 3
            else:
                raise KeyError("spin should be 'x','y','z' or 1,2,3")
            proj_data = self.spins[spin]
            proj_data /= self.spins[0]
            default_plot_kwargs = {'color':'seismic','vmin':-1,'vmax':1,'cbar_title':f"$S_{['x','y','z'][spin-1]}$ $(\\uparrow = +)$"}
            
        elif proj_sites is not None:
            if type(proj_sites) is int:
                proj_data = self.sites[proj_sites]
            elif type(proj_sites) is list or type(proj_sites) is tuple or type(proj_sites) is range:
                proj_data = np.zeros_like(self.spins[0])
                for site in proj_sites:
                    proj_data += self.sites[site]
            else:
                raise ValueError("proj_sites should be int or list/tuple/range of sites you want to project onto")
            if type(proj_sites) is range:
                proj_sites = tuple(proj_sites)
            proj_data /= self.spins[0]
            default_plot_kwargs = {'color':'afmhot_r','vmin':0,'vmax':1,'cbar_title':f"Projection onto Atomic Sites {proj_sites}"}

        else: # Defaults for if we're just plotting normal lines and no projection is happening
            default_plot_kwargs = {'color':'k','linestyle':'solid'}

        # This chunk of the code actually does the plotting
        plot_kwargs = default_plot_kwargs | (plot_kwargs or {})
        if color is None:
            color = plot_kwargs.pop('color',None)
        else:
            plot_kwargs.pop('color')
        if proj_data is None:
            for i in bands_to_plot:
                ax.plot(x_axis,self.band_energies[:,i],color=color,**plot_kwargs)
        else:
            cbar_title = plot_kwargs.pop("cbar_title")
            if mcolors.is_color_like(color):
                cmap = mcolors.LinearSegmentedColormap.from_list('trans_cmap',[mcolors.to_rgba(color,0),mcolors.to_rgba(color,1)],N=100)
            else: cmap = color
            for i in bands_to_plot:
                lines = colored_line(x_axis,self.band_energies[:,i],proj_data[:,i],ax,cmap=cmap,**plot_kwargs)
            cbar = fig.colorbar(lines)
            cbar.set_label(cbar_title)
        ax.set_xlim(0,x_axis[-1])
        if E_range is not None:
            ax.set_ylim(E_range)
        ax.set_ylabel("$\\rm E - E_F$ (eV)")

        if scaled:
            tick_locs = np.array([sum(self.line_lengths[0:i]) for i in range(self.nlines+1)])
        else:
            tick_locs = np.arange(0,self.nlines+1)
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(self.kpoint_labels)
        for tick in tick_locs:
            ax.axvline(x=tick,color="k")
        if k_range is not None: # Truncates the plot to the index of k labels that are asked for
            ax.set_xlim((tick_locs[k_range[0]],tick_locs[k_range[1]]))
        return fig, ax
    
    def plot_DOS(self,integrated=False,E_range=None,pltfigax=None,**plot_kwargs):
        if not self.subtracted:
            self.subtract_ef()
        if pltfigax is None:
            fig, ax = plt.subplots()
        else: fig, ax = pltfigax
        if integrated:
            ax.plot(self.dos_energies,self.dos_integrated,**plot_kwargs)
            ax.set_ylabel("Integrated Density of States")
        else:
            ax.plot(self.dos_energies,self.dos,**plot_kwargs)
            ax.set_ylabel("Density of States (1/eV)")
        if E_range is not None:
            ax.set_xlim(E_range)
        ax.set_xlabel("Binding Energy (eV)")
        return fig, ax
    
    def get_orbital_filling(self,atom:str,orbital:str,ef_shift:float=0):
        if atom not in self.atom_types:
            raise KeyError("The atom you asked for isn't in this material")
        
        if orbital == "s":
            data = self.dos_projected[atom][0]
        elif orbital == "p":
            data = np.sum(self.dos_projected[atom][1:4],axis=0)
        elif orbital == "d":
            data = np.sum(self.dos_projected[atom][4:9],axis=0)
        elif orbital == "f":
            data = np.sum(self.dos_projected[atom][9:16],axis=0)
        elif orbital == "total" or orbital is None:
            data = np.sum(self.dos_projected[atom],axis=0)
        elif orbital not in self.lm_labels:
            raise KeyError("That orbital name isn't in VASP")
        else:
            data = self.dos_projected[atom][self.lm_labels.index(orbital)]
        
        total_filling = np.trapz(data,self.dos_energies) # Because of the way VASP calculates orbital projections, this is what I normalize to
        ef_index = np.argmax(self.dos_energies > ef_shift)
        real_filling = np.trapz(data[:ef_index],self.dos_energies[:ef_index])
        if orbital in self.lm_labels:
            num_states = 2
        else:
            num_states = {'s':2,'p':6,'d':10,'f':14}[orbital]
        filling = real_filling/total_filling * num_states
        print(f"The filling state for the {atom} {orbital} orbital is: {filling:.3f}")
        return filling

    def plot_along_kaxis(self,axis:int,spin=None,color = None,pltfigax=None,**plotargs):
        zero_axes = [1,2,3] # If I'm plotting along a certain axis, that's the one axis that won't have all its values be 0
        if axis not in zero_axes:
            raise ValueError("Axis should be an integer corresponding to a k direction: kx=1, ky=2, kz=3")
        zero_axes.remove(axis)
        if not self.subtracted:
            self.subtract_ef()
        axis_kpoint_values = [] # Position of a k point along the chosen axis
        axis_kpoint_bands = [] # The actual eigenvalues of the bands for each k point
        axis_proj_data = []
        for i,kpoint in enumerate(self.kpoints_cartesian):
            if round(kpoint[zero_axes[0]-1],3) == 0 and round(kpoint[zero_axes[1]-1],3) == 0:
                axis_kpoint_values.append(kpoint[axis-1])
                axis_kpoint_bands.append(self.band_energies[i])
                if spin is not None:
                    axis_proj_data.append(self.spins[spin,i]/self.spins[0,i])
        axis_kpoint_values = np.array(axis_kpoint_values)
        axis_kpoint_bands = np.array(axis_kpoint_bands)
        if spin is not None:
            axis_proj_data = np.array(axis_proj_data)
        sorted_indices = np.argsort(axis_kpoint_values)
        kpoints = axis_kpoint_values[sorted_indices]
        bands = axis_kpoint_bands[sorted_indices]
        if pltfigax is None:
            fig, ax = plt.subplots()
        else: fig, ax = pltfigax

        if spin is not None:
            proj_data = axis_proj_data[sorted_indices]
            if color is None:
                color = 'seismic'
            for i in range(self.nbands):
                lines = colored_line(kpoints,bands[:,i],proj_data[:,i],ax,cmap=color,vmin=-1,vmax=1,**plotargs)
            cbar = fig.colorbar(lines)
            cbar.set_label(f"$S_{['x','y','z'][spin-1]}$ $(\\uparrow = +)$")
        else:
            if color is None:
                color = 'k'
            for i in range(self.nbands):
                ax.plot(kpoints,bands[:,i],color=color,**plotargs)
        return fig, ax

    def plot_over_arpes(self,path:list,spectrum:xr.DataArray=None,symmetrize:bool=False,mass_enhancement:float=1,ef_shift:float=0,k_center:float=0,bands_to_plot:int=None,color:str=None,proj_atom=None,proj_orbital=None,add_toplabels:bool=True,add_colorbar:bool=True,return_data=False,pltfigax=None,arpes_kwargs:dict=None,**dft_kwargs):
        """ Plots DFT over an ARPES spectrum 
        :param path: Tuple or list indicating the path you want where the first element is the center of the spectrum like ["X","M"] or [3,4] if you want to go by index
        :param spectrum: ARPES spectrum you want to plot the DFT over (pass None if you just want to plot the DFT)
        :param symmetrize: Whether you want to reflect the DFT about the center
        :param color: Color of the plotted bands. If you do orbital projection, you can put in either a color (which will be used in a transparent to opaque color map) or a matplotlib cmap
        :param ef_shift: Shift of the bands in energy (positive moves higher wrt ARPES)
        :param k_center: k value where the first element of your path is in the ARPES data
        :param mass_enhancement: Scales the DFT by 1/mass_enhancement wrt Ef
        :param proj_atom: Atom to do orbital projection onto
        :param proj_orbital: Orbital to project onto, can be s,p,d,f or specific orbital names. None or 'total' will do overall atom projection
        :param add_toplabels: Whether you want k point labels on the top of the plot
        :param return_data: Just return the band's E(k) data along the selected cut
        :param pltfigax: Tuple of matplotlib fig and ax you want to plot on like (fig,ax)
        :param arpes_kwargs: Dict of parameters to pass to the ARPES plot call
        :param dft_kwargs: Excess parameters are passed to the DFT bands plot call (like vmin,vmax for the colorbar)
        """
        if self.band_plot == False:
            raise NotImplementedError("This VASP file isn't in line mode, this function ain't built for this")
        if not self.subtracted:
            self.subtract_ef()

        if not (isinstance(path,list) or isinstance(path,tuple)):
            raise ValueError("path should be a tuple or list of two elements that are either strings or ints indicating which path you want to look along")
        if type(path[0]) != type(path[1]):
            raise ValueError("I'm not bothering to code the case where you specify a high symmetry point for one and a path index for the other, pick one")
        if isinstance(path[0],str): # If the path is specified by strings representing high symmetry points, find them 
            path_start_indices = [i for i, v in enumerate(self.kpoint_labels_raw) if v == path[0]]
            fail = True
            for index in path_start_indices:
                if index % 2 and self.kpoint_labels_raw[index - 1] == path[1]: # If the start of the path is the end of a line segment, then check the one before it and see if it matches the end of the path
                    starting_kpoint_index = self.kpoints_per_line * ((1+index) // 2) - index % 2
                    ending_kpoint_index = self.kpoints_per_line * (index // 2) - (index-1) % 2
                    fail = False
                    break
                elif not index % 2 and self.kpoint_labels_raw[index + 1] == path[1]: # If the start of the path is the start of a line segment, then check the one after it and see if it matches the end of the path
                    starting_kpoint_index = self.kpoints_per_line * ((1+index) // 2) - index % 2
                    ending_kpoint_index = self.kpoints_per_line * ((2+index) // 2) - (index+1) % 2
                    fail = False
                    break
            if fail:
                raise ValueError(f"Sorry, I couldn't find the path you asked for in the VASP file. Here's the KPOINT labels I was working with: {self.kpoint_labels}")
        elif isinstance(path[0],int): # If the path is specified by kpoint index
            if path[1] > path[0]:
                starting_kpoint_index = path[0]*self.kpoints_per_line
                ending_kpoint_index = path[1]*self.kpoints_per_line - 1
            elif path[0] > path[1]:
                starting_kpoint_index = path[0]*self.kpoints_per_line - 1
                ending_kpoint_index = path[1]*self.kpoints_per_line
            else: raise ValueError("Both path indices can't be the same, dummy")
        
        backwards = 1 - 2*(starting_kpoint_index > ending_kpoint_index) # If the cut that was asked for is in reverse order from how the kpoints are ordered in VASP, this makes sure the array is sliced backwards (should be either -1 or 1)
        band_data = self.band_energies[starting_kpoint_index:ending_kpoint_index+backwards:backwards]
        kpoint_data = self.kpoints_cartesian[starting_kpoint_index:ending_kpoint_index+backwards:backwards]
        
        kpoint_norms = np.linalg.norm(kpoint_data - kpoint_data[0],axis=1)
        if symmetrize:
            band_data = np.vstack([band_data[-1:0:-1,:],band_data])
            kpoint_norms = np.append(-kpoint_norms[-1:0:-1],kpoint_norms)

        if return_data:
            if mass_enhancement != 1 or ef_shift != 0:
                print(f"Here's the data with mass enhancement {mass_enhancement} and ef shift {ef_shift}")
            return kpoint_norms,1/mass_enhancement*(band_data+ef_shift)
        
        proj_data = None
        if proj_atom is not None:
            if proj_orbital == "s":
                proj_data = self.atom_projections[proj_atom][0]
            elif proj_orbital == "p":
                proj_data = np.sum(self.atom_projections[proj_atom][1:4],axis=0)
            elif proj_orbital == "d":
                proj_data = np.sum(self.atom_projections[proj_atom][4:9],axis=0)
            elif proj_orbital == "f":
                proj_data = np.sum(self.atom_projections[proj_atom][9:16],axis=0)
            elif proj_orbital == "total" or proj_orbital is None:
                proj_data = np.sum(self.atom_projections[proj_atom],axis=0)
            elif proj_orbital not in self.lm_labels:
                raise KeyError("That orbital name isn't in VASP")
            else:
                proj_data = self.atom_projections[proj_atom][self.lm_labels.index(proj_orbital)]
            proj_data /= self.spins[0]
            proj_data = proj_data[starting_kpoint_index:ending_kpoint_index+backwards:backwards]
            if symmetrize:
                proj_data = np.vstack([proj_data[-1:0:-1,:],proj_data])
        elif proj_orbital is not None:
            raise ValueError("Make sure to specifiy an atom to project onto")

        # Plotting section starts here -----------------------------
        if pltfigax is None:
            fig, ax = plt.subplots()
        else: fig, ax = pltfigax

        if spectrum is not None: # Plots the ARPES
            default_arpes_kwargs = {'robust':True,'add_colorbar':False,'cmap':'inferno','vmin':0}
            arpes_kwargs = default_arpes_kwargs | (arpes_kwargs or {}) # Gemini was spitting bars with this line
            spectrum.plot(ax=ax,**arpes_kwargs,zorder=0)

        if symmetrize and add_toplabels:
            secax = ax.secondary_xaxis('top')
            secax.set_ticks([k_center+kpoint_norms[0],k_center,k_center+kpoint_norms[-1]],[path[1],path[0],path[1]])
        elif add_toplabels:
            secax = ax.secondary_xaxis('top')
            secax.set_ticks([k_center,k_center+kpoint_norms[-1]],[path[0],path[1]])

        default_dft_kwargs = {'color':'k'}
        if proj_data is not None:
            default_dft_kwargs['vmax'] = 1
            default_dft_kwargs['vmin'] = 0
        dft_kwargs = default_dft_kwargs | (dft_kwargs or {})
        if color is None:
            color=dft_kwargs.pop('color')
        else:
            dft_kwargs.pop('color')

        if bands_to_plot is not None:
            if isinstance(bands_to_plot,int):
                bands_to_plot = [bands_to_plot]
            else:
                bands_to_plot = list(bands_to_plot)
        else:
            bands_to_plot = range(self.nbands)
        if proj_data is None:
            for i in bands_to_plot:
                ax.plot(kpoint_norms+k_center,1/mass_enhancement*(band_data[:,i]+ef_shift),color=color,**dft_kwargs)
        else:
            for i in bands_to_plot:
                if mcolors.is_color_like(color):
                    cmap = mcolors.LinearSegmentedColormap.from_list('trans_cmap',[mcolors.to_rgba(color,0),mcolors.to_rgba(color,1)],N=100)
                else: cmap = color
                
                lines = colored_line(kpoint_norms+k_center,1/mass_enhancement*band_data[:,i] + ef_shift,proj_data[:,i],ax,cmap=cmap,**dft_kwargs)
            if add_colorbar:
                cbar = fig.colorbar(lines)
                cbar.set_label(f"Projection onto {proj_atom} {proj_orbital if proj_orbital else ''} orbital")
        ax.set_xlabel('$\\rm k_x~(\\AA^{-1}$)')
        ax.set_ylabel('$\\rm E - E_F~(eV)$')
        return fig,ax

    def extract_band_data(self,path:list,band:int,symmetrize=True,E_range=None):
        """Wraps plot_over_arpes to return the data for one band in one array and allows energy range masking"""
        kpoints, bands_data = self.plot_over_arpes(path,symmetrize=symmetrize,return_data=True)
        data = np.column_stack((kpoints,bands_data[:,band]))
        if E_range:
            mask = (data[:, 1] > E_range[0]) & (data[:, 1] < E_range[1])
            data = data[mask]
        return data
    

def KPOINTS_Printer_zplane(kmax,Ngrid,kz_list=0,title="Fermi Grid KPOINTS"): # Will only made grids with odd numbers so gamma is included
    if Ngrid % 2 == 0:
        raise NotImplementedError("I am a dumb KPOINTS Generator and can only do odd sized grids")
    if type(kz_list) is float or type(kz_list) is int:
        kz_array = np.array([kz_list])/(2*np.pi)
    else: kz_array = np.array(kz_list)/(2*np.pi)
    print(f"{title}\n{(Ngrid**2)*len(kz_array)}\nCartesian")
    for zval in kz_array:
        for i in range(Ngrid//2+1):
            for j in range(Ngrid//2+1):
                xval = i*kmax/(Ngrid//2)/(2*np.pi)
                yval = j*kmax/(Ngrid//2)/(2*np.pi)
                if i != 0 and j != 0:
                    print(f"{xval} {yval} {zval} 1")
                    print(f"{-xval} {yval} {zval} 1")
                    print(f"{xval} {-yval} {zval} 1")
                    print(f"{-xval} {-yval} {zval} 1")
                elif i == 0 and j != 0:
                    print(f"{xval} {yval} {zval} 1")
                    print(f"{xval} {-yval} {zval} 1")
                elif i != 0 and j == 0:
                    print(f"{xval} {yval} {zval} 1")
                    print(f"{-xval} {yval} {zval} 1")
                else: 
                    print(f"{xval} {yval} {zval} 1")

def KPOINTS_Printer_yplane(kmax,Ngrid,ky_list=0,title="Fermi Grid KPOINTS"):
    if Ngrid % 2 == 0:
        raise NotImplementedError("I am a dumb KPOINTS Generator and can only do odd sized grids")
    if type(ky_list) is float or type(ky_list) is int:
        ky_array = np.array([ky_list])/(2*np.pi)
    else: ky_array = np.array(ky_list)/(2*np.pi)
    print(f"{title}\n{(Ngrid**2)*len(ky_array)}\nCartesian")
    for yval in ky_array:
        for i in range(Ngrid//2+1):
            for j in range(Ngrid//2+1):
                xval = i*kmax/(Ngrid//2)/(2*np.pi)
                zval = j*kmax/(Ngrid//2)/(2*np.pi)
                if i != 0 and j != 0:
                    print(f"{xval} {yval} {zval} 1")
                    print(f"{-xval} {yval} {zval} 1")
                    print(f"{xval} {yval} {-zval} 1")
                    print(f"{-xval} {yval} {-zval} 1")
                elif i == 0 and j != 0:
                    print(f"{xval} {yval} {zval} 1")
                    print(f"{xval} {yval} {-zval} 1")
                elif i != 0 and j == 0:
                    print(f"{xval} {yval} {zval} 1")
                    print(f"{-xval} {yval} {zval} 1")
                else: 
                    print(f"{xval} {yval} {zval} 1")

def KPOINTS_Printer_xplane(kmax,Ngrid,kx_list=0,title="Fermi Grid KPOINTS"): # Really the grid will be Ngrid + 1 in width because of the zero point
    if Ngrid % 2 == 0:
        raise NotImplementedError("I am a dumb KPOINTS Generator and can only do odd sized grids")
    if type(kx_list) is float or type(kx_list) is int:
        kx_array = np.array([kx_list])/(2*np.pi)
    else: kx_array = np.array(kx_list)/(2*np.pi)
    print(f"{title}\n{(Ngrid**2)*len(kx_array)}\nCartesian")
    for xval in kx_array:
        for i in range(Ngrid//2+1):
            for j in range(Ngrid//2+1):
                yval = i*kmax/(Ngrid//2)/(2*np.pi)
                zval = j*kmax/(Ngrid//2)/(2*np.pi)
                if i != 0 and j != 0:
                    print(f"{xval} {yval} {zval} 1")
                    print(f"{xval} {-yval} {zval} 1")
                    print(f"{xval} {yval} {-zval} 1")
                    print(f"{xval} {-yval} {-zval} 1")
                elif i == 0 and j != 0:
                    print(f"{xval} {yval} {zval} 1")
                    print(f"{xval} {yval} {-zval} 1")
                elif i != 0 and j == 0:
                    print(f"{xval} {yval} {zval} 1")
                    print(f"{xval} {-yval} {zval} 1")
                else:
                    print(f"{xval} {yval} {zval} 1")


def KPOINTS_Printer_plane(kmax,Ngrid,plane_vector=(0,0,1),kperp_list=0,shift:np.ndarray=None,title="Fermi Grid KPOINTS",filepath:str=None):
    if Ngrid % 2 == 0:
        raise NotImplementedError("I am a dumb KPOINTS Generator and can only do odd sized grids")
    if type(kperp_list) is float or type(kperp_list) is int:
        kperp_array = np.array([kperp_list])
    else: kperp_array = np.array(kperp_list)
    nkpoints = Ngrid**2 * len(kperp_array)
    # First, I get all of the points in the planes perp to (001)
    kparallel_vals = np.linspace(-kmax,kmax,Ngrid,endpoint=True)
    k1,k2,k3 = np.meshgrid(kparallel_vals,kparallel_vals,kperp_array)
    kpoints = np.stack([k1,k2,k3],axis=-1).reshape(nkpoints,3)
    
    # Now, I take all of the points oriented around (001) then rotate to whatever unit vector is provided
    if not all([plane_vector[i] == (0,0,1)[i] for i in range(3)]): # If the points are already perp to (0,0,1), no need to rotate
        # Getting theta and phi of the unit vector then doing R_y(theta).R_z(phi) on each point
        theta = np.arccos(plane_vector[2]/np.linalg.norm(plane_vector))
        phi = np.sign(plane_vector[1])*np.arccos(plane_vector[0]/np.sqrt(plane_vector[0]**2+plane_vector[1]**2))
        #print(theta,phi)
        rot_matrix = np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]]) @ np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        #print(np.linalg.det(rot_matrix))
        for i in range(len(kpoints)):
            kpoints[i] = rot_matrix @ kpoints[i]
        kpoints = np.round(kpoints,10)
    if shift is not None:
        kpoints += shift
    print(kpoints)
    kpoints = kpoints / (2*np.pi) # For some dumbass reason, VASP scales everything up by 2pi
    # Now to actually print the KPOINTS file to the designated filepath:
    if filepath is not None:
        if os.path.isdir(filepath):
            if filepath.endswith(os.sep):
                filepath = filepath + os.sep + "KPOINTS"
            else: filepath = filepath + "KPOINTS"
        with open(filepath,"w",newline='\n') as file:
            file.write(f"{title}\n{len(kpoints)}\nCartesian\n")
            for i in range(len(kpoints)):
                file.write(f"{kpoints[i,0]} {kpoints[i,1]} {kpoints[i,2]} 1\n")
        return kpoints
    else: # If no file path is given, just print it out to the terminal
        print(f"{title}\n{len(kpoints)}\nCartesian")
        for i in range(len(kpoints)):
            print(f"{kpoints[i,0]} {kpoints[i,1]} {kpoints[i,2]} 1")
        return None


class Fermi2D_Plotter():
    # Takes in a parser to get data out of and gets variables as needed make 2D Fermi Surface Plots
    # By default, it assumes you want a vaspout.h5 file but you can swap it out for any other parser that has the same functions
    def __init__(self,directory,parser:str='vasph5',Ef:float=None,z_axis=None):
        """
        :param directory: Directory where the DFT output is
        :param parser: What parser you want to use, current options are: 'vasph5' or 'vasptxt' (which uses the old EIGENVAL file)
        :param Ef: Custom Ef to use (overrides the one from the DFT run)
        :param z_axis: Unit vector in the VASP coordinate system for what you want your new z axis to be
        """
        self.directory = directory
        self.ef = Ef
        if parser is None:
            print("Welp it seems you haven't given me any parser so load the data yourself if you wanna make plots")
            return
        elif isinstance(parser,str):
            parser_dict = {'vasph5':vaspouth5_Parser,'vaspouth5':vaspouth5_Parser,'EIGENVAL':EIGENVAL_Parser,'eigenval':EIGENVAL_Parser,'vasptxt':EIGENVAL_Parser}
            self.parser = parser_dict[parser](directory,Ef=Ef,rotation=z_axis)
            if not self.parser.subtracted:
                self.parser.subtract_ef()
            self.ef = self.parser.e_fermi
        elif not parser.initialzed:
            self.parser = parser(directory,Ef=Ef,rotation=z_axis)
            if not self.parser.subtracted:
                self.parser.subtract_ef()
            self.ef = self.parser.e_fermi
        else: # This is here in case you've already initialized the parser, this probably isn't needed but whatever
            self.parser = parser
            if not self.parser.subtracted:
                self.parser.subtract_ef()
            self.ef = self.parser.e_fermi

    def __getattr__(self, name):
        # Grabs the different xarrays only when the plotter needs it to save a little time
        if name == "occupations":
            self.occupations = self.parser.get_occupation_xarray()
            return self.occupations
        if name == "energy":
            self.energy = self.parser.get_bands_xarray()
            return self.energy
        
        if name == "plotFermi2D_kxplane": # For compatibility with old notebooks from before I combined these functions
            def plotFermi2D_kxplane(*args,**kwargs):
                if "kx" in kwargs:
                    kwargs["k_perp"] = kwargs.pop("kx")
                return self.plotFermi2D('x',*args,**kwargs)
            return plotFermi2D_kxplane
        if name == "plotFermi2D_kyplane":
            def plotFermi2D_kyplane(*args,**kwargs):
                if "ky" in kwargs:
                    kwargs["k_perp"] = kwargs.pop("ky")
                return self.plotFermi2D('y',*args,**kwargs)
            return plotFermi2D_kyplane
        if name == "plotFermi2D_kzplane":
            def plotFermi2D_kzplane(*args,**kwargs):
                if "kz" in kwargs:
                    kwargs["k_perp"] = kwargs.pop("kz")
                return self.plotFermi2D('z',*args,**kwargs)
            return plotFermi2D_kzplane
        
        if isinstance(self.parser,EIGENVAL_Parser):
            raise NotImplementedError("EIGENVAL Parser can't be used to get spins or atom projections")
        if name == "sx":
            self.sx = self.parser.get_spin_xarray(1)
            return self.sx
        if name == "sy":
            self.sy = self.parser.get_spin_xarray(2)
            return self.sy
        if name == "sz":
            self.sz = self.parser.get_spin_xarray(3)
            return self.sz
        if name == "projections":
            self.projections = self.parser.get_atom_projections_xarray()
            return self.projections
        if name == "lm_labels":
            self.lm_labels = self.parser.lm_labels
            return self.lm_labels
        if name == "atoms":
            self.atoms = self.parser.atom_types
            return self.atoms

    @staticmethod
    def interpolate_xarray(xarray:xr.DataArray,N_interp:int=5,dims_to_interp=None): 
        """This function interpolates an xarray, ensuring the original points are in it

        :param xarray: xr.DataArray to be interpolated
        :param N_interp: The number of points to be added between each point
        :param dims_to_interp: List of dimensions to interpolate over if you don't want them all
        """

        if dims_to_interp is None: # Get all of the coordinates of the xarray and if any coordinates have only one value, skip them
            #dims_to_interp = list(xarray.coords.keys())
            dims_to_interp = [dim for dim in xarray.coords if xarray.coords[dim].size > 1]
        elif isinstance(dims_to_interp,str):
            dims_to_interp = [dims_to_interp]
        new_coords = {}
        for dim in dims_to_interp:
            original_points = xarray.coords[dim].values
            new_points = np.zeros(len(original_points)+N_interp*(len(original_points)-1))
            for i in range(len(original_points)-1):
                new_points[i*(N_interp+1):(i+1)*(N_interp+1)+1] = np.linspace(original_points[i],original_points[i+1],N_interp+2,endpoint=True)
            new_coords[dim] = new_points
        new_xarray = xarray.interp(new_coords,method='cubic')
        return new_xarray

    def plotFermi2D(self,k_plane:str='kz',k_perp:float=0,fermi:float=0,color="k",interp:int=3,spin=None,fermi_bands:Optional[List[int]]=None,spin_texture=False,st_interp:int=0,atom:str=None,orbital:str=None,proj_cmap=None,add_colorbar=True,pltfigax=None):
        """Plots a 2D slice of a Fermi Surface using a regularly spaced xarray with dims 'kx','ky','kz', 'band'

        :param k_plane (str): String indicating the k coordinate the fermi surface is perpendicular to, i.e. 'kx' if you want the slice at kx=0. Accepts in the form of 'ky', 'y' or 2
        :param k_perp (float): k coordinate of desired 2D slice, i.e. 0.7 if you want the kz=0.7 A^-1 slice
        :param fermi (float): Desired energy contour wrt Ef, i.e. -0.25 if you want the E-Ef = -0.25 contour
        :param color (str): Color of fermi surface contours
        :param interp (int): Interpolation factor, higher numbers take longer but give smoother contours and spin projection plots
        :param spin (str): Spin component to plot on top of fermi surface, in the format 'x','y','z'
        :param fermi_bands (list): List of bands to plot the fermi surface contours for (will be all bands at desired E-Ef if nothing given)
        :param spin_texture (bool): Give True to plot in-plane spin texture for only one band (specify this band in fermi_bands)
        :param st_interp (int): Interpolation factor for spin texture plotting
        :param atom (str): Atom to project onto, use periodic table symbols
        :param orbital (str): Orbital to project into, can be 's','p','d','f' or any specific orbital like 'dxy'. None or 'total' sums all orbitals
        :param proj_cmap (str): Color map for spin or orbital projection
        :param pltfigax: Matplotlib figure and axis to plot onto, given in the tuple (fig, ax)
        """
        # k1 is the horizontal axis of the plot, k2 is the vertical axis, and k3 is perpendicular to the plot
        if k_plane in ['x','kx',1]: 
            k1,k2,k3 = 'ky','kz','kx'
        elif k_plane in ['y','ky',2]:
            k1,k2,k3 = 'kx','kz','ky'
        elif k_plane in ['z','kz',3]:
            k1,k2,k3 = 'kx','ky','kz'
        else:
            raise ValueError("k_plane should be the k coordinate your 2D slice is perpendicular to, like 'kx','ky','kz'")
        k3_slice = self.energy.sel({k3:k_perp}, method="nearest")
        if fermi_bands is None: # If the function hasn't been given what bands are to be plotted, find the ones at the fermi level
            fermi_bands = []
            for band_num, band in k3_slice.groupby("band"):
                if band.max().item() > fermi and band.min().item() < fermi:
                    fermi_bands.append(band_num)
        if len(fermi_bands) > 0:
            print("Bands at the Fermi Level to be Plotted:",[int(band) for band in fermi_bands])
        else:
            print("No bands at the selected fermi level :(")
        if pltfigax is None:
            fig,ax = plt.subplots()
        else: fig,ax = pltfigax
        ax.set_aspect(aspect="equal",adjustable="box")
        for key in list(ax.spines.keys()):
            ax.spines[key].set_zorder(5) # Makes sure the axes are on top after all my antics
        spin_coords = []
        print(f"Making Fermi Plot at E-Ef = {round(fermi,3)} eV and {k3} = {float(k3_slice.coords[k3])} A^-1")
        for band_num in fermi_bands:
            interpolated_band = self.interpolate_xarray(k3_slice.sel(band=band_num,method="nearest"),interp)
            contour = interpolated_band.plot.contour(x=k1,y=k2,levels=[fermi],colors=color,ax=ax,linestyles='solid',zorder=2)
            spin_coords.append(np.vstack(contour.allsegs[0])) # Gets the points used to make the contour plot to plot spin there
        if spin is not None:
            print(f"Starting spin plotting for spin component {spin}")
            if spin in ['x','sx',1]:
                spin = 'x'
            elif spin in ['y','sy',2]:
                spin = 'y'
            elif spin in ['z','sz',3]:
                spin = 'z'
            else: 
                raise ValueError(f"Invalid spin component, must be 'x','y','z', 1, 2, or 3")
            spins = getattr(self,'s'+spin)
            spin_data = []
            spins_slice = spins.sel({k3:k_perp}, method = "nearest")
            for band_num, spin_locs in zip(fermi_bands,spin_coords):
                spin_data_band = []
                spin_band = spins_slice.sel(band=band_num,method='nearest')
                for point in spin_locs:
                    spin_data_band.append(float(spin_band.interp({k1:point[0],k2:point[1]},method='linear')))
                spin_data.append(spin_data_band)
            for i in range(len(spin_data)):
                ax.scatter(spin_coords[i][:,0],spin_coords[i][:,1],s=10,c=spin_data[i],cmap='seismic',vmin=-1,vmax=1,zorder=1)
            if add_colorbar:
                cbar = fig.colorbar(cm.ScalarMappable(cmap=cm.seismic,norm=mcolors.Normalize(vmin=-1,vmax=1)),ax=ax)
                cbar.set_label(f"$S_{spin}$ $(+ = \\uparrow)$")
        if spin_texture and len(fermi_bands) == 1: # Plots vector plot for spin components in plane of the cut
            print(f"Starting in plane spin texture plotting for S{k1[1]} and S{k2[1]} for band {fermi_bands[0]}")
            s1_slice = getattr(self,'s'+k1[1]).sel({'band':fermi_bands[0], k3:k_perp},method='nearest')
            s2_slice = getattr(self,'s'+k2[1]).sel({'band':fermi_bands[0], k3:k_perp},method='nearest')
            if st_interp > 0:
                s1_slice = self.interpolate_xarray(s1_slice,st_interp)
                s2_slice = self.interpolate_xarray(s2_slice,st_interp)
            X,Y = np.meshgrid(s1_slice.coords[k1],s1_slice.coords[k2])
            ax.quiver(X,Y,s1_slice.T,s2_slice.T,angles='uv',scale_units='xy')
        elif spin_texture:
            print("Warning: Number of fermi bands to be plotted must be 1 to plot spin texture, skipping")
        if atom in self.atoms and spin is None:
            print(f"Starting atomic orbital projection plotting for {atom} {orbital} orbital")
            if orbital in self.lm_labels:
                proj_slice = self.projections[atom].sel({k3: k_perp},method='nearest').sel(orbital=orbital)
                if orbital == "s":
                    proj_cmap_default = "Blues"
                else:
                    proj_cmap_default = "Reds"
            elif orbital == "p":
                proj_slice = self.projections[atom].sel({k3: k_perp},method='nearest').sel(orbital=self.lm_labels[1:4]).sum(dim='orbital')
                proj_cmap_default = "Purples"
            elif orbital == "d" and len(self.projections[atom]['orbital']) > 4:
                proj_slice = self.projections[atom].sel({k3: k_perp},method='nearest').sel(orbital=self.lm_labels[4:9]).sum(dim='orbital')
                proj_cmap_default = "Greens"
            elif orbital == "f" and len(self.projections[atom]['orbital']) > 9:
                proj_slice = self.projections[atom].sel({k3: k_perp},method='nearest').sel(orbital=self.lm_labels[9:16]).sum(dim='orbital')
                proj_cmap_default = "Oranges"
            elif orbital == "total" or orbital is None:
                proj_slice = self.projections[atom].sel({k3: k_perp},method='nearest').sum(dim='orbital')
                proj_cmap_default = "Reds"
            else: 
                raise KeyError("The requested orbital for projection doesn't exist")
            if proj_cmap is None:
                proj_cmap = cm.get_cmap(proj_cmap_default)
            elif mcolors.is_color_like(proj_cmap):
                proj_cmap = mcolors.LinearSegmentedColormap.from_list('trans_cmap',[mcolors.to_rgba(proj_cmap,0),mcolors.to_rgba(proj_cmap,1)],N=100)
            else:
                proj_cmap = cm.get_cmap(proj_cmap)
            proj_data = []
            for band_num, proj_locs in zip(fermi_bands,spin_coords):
                proj_data_band = []
                proj_band = proj_slice.sel(band=band_num,method='nearest')
                for point in proj_locs:
                    proj_data_band.append(float(proj_band.interp({k1:point[0],k2:point[1]},method='linear')))
                proj_data.append(proj_data_band)
            for i in range(len(proj_data)):
                ax.scatter(spin_coords[i][:,0],spin_coords[i][:,1],s=10,c=proj_data[i],cmap=proj_cmap,vmin=0,vmax=1,zorder=3)
            if add_colorbar:
                cbar = fig.colorbar(cm.ScalarMappable(cmap=proj_cmap,norm=mcolors.Normalize(vmin=0,vmax=1)),ax=ax)
                cbar.set_label(f"{atom} {orbital} Orbital Projection")
        elif atom is not None and atom not in self.atoms:
            raise KeyError("The requested atom for projection isn't in this material")
        elif atom is not None and spin is not None:
            print("Warning: Can't plot atom projection because spin projection has already been plotted, skipping")
        ax.set_xlabel(f"$\\rm {'k_'+k1[1]}$ (Å$^{{-1}}$)")
        ax.set_ylabel(f"$\\rm {'k_'+k2[1]}$ (Å$^{{-1}}$)")
        return fig,ax
    
    def orbital_breakdown(self,k_plane='z',k_perp=0,fermi=0,band_color='k',interp=5,atom=None):
        if atom not in self.atoms:
            raise KeyError("The requested atom for the orbital breakdown is not in this material")
        fig,axs = plt.subplots(2,2,figsize=(11.5,9.5))
        fig,axs[0,0] = self.plotFermi2D(k_plane,k_perp,fermi,band_color,interp,atom=atom,orbital='s',pltfigax=(fig,axs[0,0]))
        axs[0,0].set_title(f"{atom} s Orbital")
        fig,axs[0,1] = self.plotFermi2D(k_plane,k_perp,fermi,band_color,interp,atom=atom,orbital='p',pltfigax=(fig,axs[0,1]))
        axs[0,1].set_title(f"{atom} p Orbital")
        if len(self.projections[atom]['orbital']) > 4:
            fig,axs[1,0] = self.plotFermi2D(k_plane,k_perp,fermi,band_color,interp,atom=atom,orbital='d',pltfigax=(fig,axs[1,0]))
            axs[1,0].set_title(f"{atom} d Orbital")
        if len(self.projections[atom]['orbital']) > 9:
            fig,axs[1,1] = self.plotFermi2D(k_plane,k_perp,fermi,band_color,interp,atom=atom,orbital='f',pltfigax=(fig,axs[1,1]))
            axs[1,1].set_title(f"{atom} f Orbital")
        else:
            fig,axs[1,1] = self.plotFermi2D(k_plane,k_perp,fermi,band_color,interp,atom=atom,orbital='total',pltfigax=(fig,axs[1,1]))
            axs[1,1].set_title(f"{atom} total")
        fig.suptitle(f"Atomic Orbital Projection Breakdown for {atom} at $k_{k_plane} = {k_perp}$ and $E-E_f = {fermi} eV$",y=0.925)
        return fig,axs
    
    def plot_bands_along_kaxis(self,axis='z',interp=5,spin=None,pltfigax=None,k1_offset=0,k2_offset=0,bands_to_plot=None):
        if axis in ['x','kx',1]:
            k1,k2,k3 = 'ky','kz','kx'
        elif axis in ['y','ky',2]:
            k1,k2,k3 = 'kx','kz','ky'
        elif axis in ['z','kz',3]:
            k1,k2,k3 = 'kx','ky','kz'
        else:
            raise ValueError("axis should be the k axis you want to plot along like 'kx','ky','kz'")
        band_data = self.interpolate_xarray(self.energy.sel({k1:k1_offset,k2:k2_offset},method='nearest'),interp)
        
        if spin is not None:
            if spin in ['x','sx',1]:
                spin = 'sx'
            elif spin in ['y','sy',2]:
                spin = 'sy'
            elif spin in ['z','sz',3]:
                spin = 'sz'
            else:
                raise ValueError("spin should be the spin component you want to plot along like 'sx','sy','sz'")
            spins = self.interpolate_xarray(getattr(self,spin).sel({k1:k1_offset,k2:k2_offset},method='nearest'),interp)
        
        if pltfigax is None:
            fig, ax = plt.subplots()
        else: fig, ax = pltfigax
        if bands_to_plot is None:
            bands_to_plot = range(1,self.parser.nbands+1)
        if spin is None:
            for i in bands_to_plot:
                band_data.sel(band=i).plot(color='k',ax=ax)
        else:
            for i in bands_to_plot:
                lines = colored_line(band_data[k3].values,band_data.sel(band=i).values,spins.sel(band=i).values,ax=ax,vmin=-1,vmax=1,cmap='seismic')
            cbar = fig.colorbar(lines)
            cbar.set_label(f"$S_{spin[1]}$ $(+ = \\uparrow)$")
        return fig, ax
    
    def get_effective_mass(self,band,axis='z',center=0,window=0.2,plot_fit=False,DFT_interp:int=0,kx=0,ky=0,kz=0):
        hbar = 6.582e-16 # ev*s
        c = 2.998e8 # m/s
        def parabola(k,a,k0,e0):
            return a*(k-k0)**2 + e0
        if axis == 'z' or axis == 3:
            band_data = self.energy.sel(kx=kx,ky=ky,band=band,method='nearest').sel(kz=slice(-window+center,window+center))
            if kz: print("Warning you gave me a kz value, but I'm ignoring it since I'm looking along kz")
        elif axis == 'y' or axis == 2:
            band_data = self.energy.sel(kx=kx,kz=kz,band=band,method='nearest').sel(ky=slice(-window+center,window+center))
            if ky: print("Warning you gave me a ky value, but I'm ignoring it since I'm looking along ky")
        elif axis == 'x' or axis == 1:
            band_data = self.energy.sel(ky=ky,kz=kz,band=band,method='nearest').sel(kx=slice(-window+center,window+center))
            if kx: print("Warning you gave me a kx value, but I'm ignoring it since I'm looking along kx")
        else:
            raise ValueError("Given axis should be x,y,z or 1,2,3")
        if DFT_interp:
            band_data = self.interpolate_xarray(band_data,DFT_interp)
        mass_parameters, cov_matrix = curve_fit(parabola,band_data.coords[band_data.dims[0]].values,band_data.values,[1,center+window/5,0])
        m = (hbar**2/(2*mass_parameters[0]))*(1e20) * c**2 # ev/c^2
        if plot_fit:
            fig, ax = plt.subplots()
            band_data.plot(label='DFT',ax=ax)
            band_data = self.interpolate_xarray(band_data)
            ax.plot(band_data.coords[band_data.dims[0]].values,parabola(band_data.coords[band_data.dims[0]].values,mass_parameters[0],mass_parameters[1],mass_parameters[2]),label='fit')
            ax.set_title("Band Parabola Fit")
            ax.legend()
        delta_m = np.sqrt(cov_matrix[0,0]) * (hbar**2/(2*(mass_parameters[0])**2))*(1e20) * c**2
        print(f"Estimated Band Mass: {m/511e3:3f} ± {delta_m/511e3:3e} m_e")
        return m/511e3,mass_parameters[1],mass_parameters[2]

def pymatgen_slab_generator(bulk_structure, output_prefix="./slabs_output", miller_index=(0,0,1), min_slab_size=10,
                             vacuum_size=20, selective_dynamics=None,slab_choice=None,symmetrize=False, layer_tol=0.1):
    try:
        from pymatgen.core.surface import SlabGenerator
        from pymatgen.core import Structure
        from pymatgen.io.vasp import Poscar
        from pathlib import Path
    except:
        raise ImportError("This function requires pymatgen and pathlib. Go install them.")

    outdir = Path(output_prefix)
    outdir.mkdir(parents=True, exist_ok=True)

    def cluster_z(z_values, tol):
        """Group z-coords into layers using a distance tolerance. 
        Matches pymatgen ftol=0.1 Ang default)."""
        z_sorted = sorted(z_values)
        clusters = [[z_sorted[0]]]
        for z in z_sorted[1:]:
            if z - clusters[-1][-1] <= tol:
                clusters[-1].append(z)
            else:
                clusters.append([z])
        return [np.mean(c) for c in clusters], clusters

    # make the slwabs based on input requirements
    s = Structure.from_file(bulk_structure, sort=True)
    sg = SlabGenerator(initial_structure=s, miller_index=miller_index,
                        min_slab_size=min_slab_size, min_vacuum_size=vacuum_size,
                        center_slab=True)
    slabs = sg.get_slabs(symmetrize=symmetrize)

    print(f"Found {len(slabs)} possible slabs. You chose slab numbers: {slab_choice}. Applying selective dynamics (if specified) and printing POSCAR.")

    if slab_choice == None:
        desired_slabs = slabs
    else:
        desired_slabs = []
        for idx in slab_choice:
            desired_slabs.append(slabs[idx])

    for i, slab in enumerate(desired_slabs):
        all_z_coords = [site.coords[2] for site in slab]
        layer_centers, clusters = cluster_z(all_z_coords, layer_tol)
        n_layers = len(layer_centers)
        slab_thickness = max(all_z_coords) - min(all_z_coords)
        vacuum_thickness = slab.lattice.c - slab_thickness

        if vacuum_thickness < 10:
            print(f"You just made a slab with only {vacuum_thickness:.2f} Ang. of vacuum. I hope you know what you're doing...")
            print("If you're a noob, try increasing vacuum_size to get at least 20 Ang. of space, otherwise, proceed with caution.")

        print(f"Generated slab {i} with {n_layers} layers, that is {slab_thickness:.3f} Ang. thick")
        print(f"The vacuum layer is {vacuum_thickness:.2f} Ang. thick")
        print(f"symmetric (top/bottom surfaces equivalent): {slab.is_symmetric()}")

        if selective_dynamics:
            num_layers_to_relax = selective_dynamics
            if n_layers <= (num_layers_to_relax * 2):
                print("Warning: Slab is thinner than the requested relaxation layers. All atoms will be relaxed.")
                relax_clusters = clusters
            else:
                relax_clusters = clusters[:num_layers_to_relax] + clusters[-num_layers_to_relax:]
            relax_z_values = set(z for cluster in relax_clusters for z in cluster)

            sel_dyn = []
            for site in slab:
                if site.coords[2] in relax_z_values:
                    sel_dyn.append([True, True, True]) #allow for total 3d relaxation
                else:
                    sel_dyn.append([False, False, False]) #freeze the site, not on the surface
            
            slab.add_site_property("selective_dynamics", sel_dyn)
            slab = slab.get_sorted_structure()

            slab.to(filename=outdir / f"POSCAR_slab{i}_sel_{n_layers}layers.poscar", fmt="poscar")
            print(f"I cast: selective dynamics on the top and bottom {num_layers_to_relax} layers.")
        else:
            slab = slab.get_sorted_structure() # sorts by z while keeping species together for ease of 
            slab.to(filename=outdir / f"POSCAR_slab{i}", fmt="poscar")
        print("~~~~~~~~~~~~~~~~~ Done with this slab ~~~~~~~~~~~~~~~")
#Examples:
r''' directory = r"C:\Users\ajbal\OneDrive - UCB-O365\Dessau Research\VASP Data\TaAs\bands"
    plotter = Fermi2D_Plotter(directory)
    # Example for plotting sz projection onto the bands at kz=0, E-Ef=-0.2
    fig, ax = plotter.plotFermi2D('z',k_perp=0,fermi=-0.2,spin='z')
    plt.title("2D -0.2 eV Surface and Sz projection at $k_{z}$ = 0")

    #Example for plotting in plane spin texture of a band as well as that band's 2d fermi surface at E-Ef=-0.5 
    fig, ax = plotter.plotFermi2D('z',k_perp=0,fermi=-0.5,spin='z',fermi_bands=[148],spin_texture=True)
    plt.title("Sx Sy Spin Texture and -0.5 eV Surface at $k_{z}$ = 0 for Band 148")

    #Example for plotting Cu pz orbital projection onto the fermi surface
    fig, ax = plotter.plotFermi2D('z',k_perp=0,fermi=-0.5,atom="Cu",orbital="pz")
    plt.title("Cu d Projection onto -0.5 eV Surface at $k_{z}$ = 0")

    #Breakdown by orbital of an atom projection
    fig, axs = plotter.orbital_breakdown(k=0,plane='z',fermi=-0.5,atom="Cu")
    
    # You can also specify a figure and axis for the plot if you want to make your own subplots thing
    fig2,ax2 = plt.subplots()
    fig2, ax2 = plotter.plotFermi2D_kzplane(kz=0,fermi=-0.7,spin='x',pltfigax=(fig2,ax2))
    plt.title("2D -0.7 eV Surface and Sx projection at $k_{z}$ = 0")'''