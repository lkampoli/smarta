# Copyright (C) 2020  Pietro Parodi
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


#############################################################################
# SMARTA Simulation script                                                  #
#############################################################################

from rarfunc import *

mpicomm = Comm()
universe = Universe()

rprint(mpicomm, 'Loading mesh...')

#############################################################################
# Import the mesh file(s) and add it to the universe                        #
#                                                                           #
# msh = mesh.Mesh.from_file(filename)                                       #
#   Loads the STL mesh from file.                                           #
#       - msh is the loaded mesh object                                     #
#       - filename is the path and filename to the *.stl to load            #
#                                                                           #
# universe.add(Comm, msh, ID, type, rho = 0)                                #
#   Adds the mesh to the universe.                                          #
#       - msh is the loaded mesh object                                     #
#       - Comm is an object of the Comm class (MPI communicator info)       #
#       - ID is the surface group identifier (positive integer)             #
#       - type is the surface group type ('wall', 'inlet', 'outlet')        #
#       - rho is the surface group reflectivity (0.0 - 1.0)                 #
#############################################################################

#############################################################################
# Mesh of a sphere in a cylinder acting as the flow source                  #
#############################################################################
# folder = './mesh_disks/'
# universe.add(mpicomm, folder+'disk_target.stl', 0, 'wall') 
# universe.add(mpicomm, folder+'disk_source.stl',   1,'inlet')


#############################################################################
# Mesh of Clausing's cylinder for transmission probability calculation      #
#############################################################################
# folder = './mesh_cylinder/'
# universe.add(mpicomm, folder+'inlet_long.stl',    1,  'inlet'         )
# universe.add(mpicomm, folder+'outlet_long.stl',   2, 'outlet', rho = 0)
# universe.add(mpicomm, folder+'cylinder_long.stl', 0,   'wall'         )


#############################################################################
# Mesh of the RAM-EP intake                                                 #
#############################################################################
# folder = './mesh_intake_smoother/'
# universe.add(mpicomm, folder+'wall_better_scaled.stl', 0,   'wall'         )
# universe.add(mpicomm, folder+'inlet_better.stl',       1,  'inlet'         )
# universe.add(mpicomm, folder+'outlet.stl',             2, 'outlet', rho = 0)

#############################################################################
# Mesh of a hemisphere                                                      #
#############################################################################
folder = './mesh_concave/'
universe.add(mpicomm, folder+'hemisphere.stl', 0, 'wall') 
universe.add(mpicomm, folder+'source.stl',   1,'inlet')


#############################################################################
# Done importing the mesh, initialize it.                                   #
#                                                                           #
# universe.init(Comm)                                                       #
#   Initializes the arrays in universe.                                     #
#       - Comm is an object of the Comm class (MPI communicator info)       #
#############################################################################

universe.init(mpicomm)

rprint(mpicomm, 'Loaded a total of {N} triangles.'.format(N=universe.N))

mpicomm.comm.Barrier()

#############################################################################
# Assign the properties to surface groups.                                  #
#                                                                           #
# universe.prop(ID, n = 0, T = 0, S = 0, uhat = [0, 0, 0], m = 0)           #
#   Set the properties of the gas in the reservoir of a group of surfaces.  #
#       - ID is the surface group identifier (positive integer)             #
#       - n is the number density of the gas in the reservoir               #
#       - T is the temperature of the gas in the reservoir                  #
#       - S is the speed ratio of the gas in the reservoir                  #
#       - uhat is the unit! vector of the gas velocity in the reservoir     #
#       - m is the molecular mass of the gas in the simulation              #
#############################################################################


#############################################################################
# Sphere in a cylinder acting as the flow source                            #
#############################################################################
universe.prop(0, T=300.0)
universe.prop(1, S=2.0, n=1e16, T=800, m=4e-26, uhat=[0.0,-1.0,0.0])


#############################################################################
# Clausing's cylinder for transmission probability calculation              #
#############################################################################
# universe.prop(0, T=300.0)
# universe.prop(1, S=1.0, n=1e16, T=800, m=4e-26, uhat=[0.0,1.0,0.0])


#############################################################################
# RAM-EP intake                                                             #
#############################################################################
# universe.prop(0, T=300.0)
# universe.prop(1, S=11.3, n=6.695e13, T=800, m=2.65e-26, uhat=[1.0,0.0,0.0])

mpicomm.comm.Barrier()

#############################################################################
# Compute view factor matrix or load from file                              #
#############################################################################
LOADFMATRIX = False
filename = '/nobackup/st/parodi/Fmatrix.npy'

if LOADFMATRIX == False:
    rprint(mpicomm, 'Computing view factors matrix...')
    tic(mpicomm)
    F = view_factors(mpicomm, universe)
    toc(mpicomm, 'View factor matrix computation')
else:
    if mpicomm.rank == 0:
        F = np.load(filename)
    else:
        F = None

rprint(mpicomm, 'Maximum value in F matrix is:  {val}'.format(val=np.max(F)))
rprint(mpicomm, 'Minimum value in F matrix is:  {val}'.format(val=np.min(F)))

#############################################################################
# Saving of view factor matrix to file for later use                        #
# Caution: could be multiple GB!                                            #
#############################################################################
SAVEFMATRIX = False
filename = '/nobackup/st/parodi/Fmatrix.npy'

if SAVEFMATRIX == True:
    if mpicomm.rank == 0:
        np.save(filename, F)


#############################################################################
# Show a map of the view factor matrix                                      #
# Caution: may take some time                                               #
#############################################################################
SHOWFIMAGE = False

if SHOWFIMAGE == True:
    if mpicomm.rank == 0:
        plt.imshow(np.log10(np.clip(F, 1e-4, np.max(F))), interpolation='none', cmap='binary')
        plt.colorbar()
        plt.show()

#############################################################################
# Calculation of matrices F1 and F2                                         #
#############################################################################

rprint(mpicomm, 'Computing F1 matrix...')

F1 = compute_F1(mpicomm, universe, F)

rprint(mpicomm, 'Completed.')

F2 = F

#############################################################################
# Calculation of matrix M                                                   #
#############################################################################

rprint(mpicomm, 'Computing M matrix...')
tic(mpicomm)

M = compute_M(mpicomm, universe)

toc(mpicomm, 'M matrix computation')
rprint(mpicomm, 'Maximum value in M matrix is:  {val}'.format(val=np.max(M)))


rprint(mpicomm, 'Computing E matrix...')

E = compute_E(mpicomm, universe)

#############################################################################
# Solution of the linear system with np.linalg.solve (alternatives shown)   #
#############################################################################

if mpicomm.rank == 0:
    tic(mpicomm)
    b = np.dot(M * F2, E)
    print('Solving the linear matrix equation...')
    try:
        B = np.linalg.solve(F1, b)
        # B, lstsqresiduals, lstsqrank, lstsqs = np.linalg.lstsq(F1, b)
        # B, rnorm = nnls(F1, b)
    except np.linalg.LinAlgError as e:
        print(str(e))
        raise
    toc(mpicomm, 'Solving the linear matrix equation')
else:
    B = None
mpicomm.comm.Barrier()


if mpicomm.rank == 0:
    bcheck = np.dot(F1, B)
    print('Completed. Checking if solution is exact: {res}'.format(res = np.allclose(bcheck, b)))



#############################################################################
# Computation of total flux through a surface group                         #
#                                                                           #
# tot_flux(Comm, Universe, B, ID)                                           #
#   Calculates the total flux.                                              #
#       - Comm is an object of the Comm class (MPI communicator info)       #
#       - Universe is an object of the Universe class (mesh and gas info)   #
#       - ID is the int identifier of the group of surfaces to calculate    #
#         the total flux on                                                 #
#       - B is the surface flux vector                                      #
#############################################################################
    
if mpicomm.rank == 0:
    fout = tot_flux(universe, B, 0)

# Some example calculations for the Clausing factor (or collection efficiency)
if mpicomm.rank == 0:
    A = np.pi*0.5**2
    fin = A * flux_hyper(universe.n[1], universe.T[1], universe.m, universe.S[1])
    W = fout/fin
    print('Clausing transmission factor W is: {W}'.format(W=W))


#############################################################################
# Computation of pressure forces                                            #
#                                                                           #
# px, py, pz, p = compute_pressure(Comm, Universe, F, B)                    #
#   Calculates the pressure force components on surfaces.                   #
#       - Comm is an object of the Comm class (MPI communicator info)       #
#       - Universe is an object of the Universe class (mesh and gas info)   #
#       - F is the view factor matrix                                       #
#       - B is the surface flux vector                                      #
#       - px, py, pz are the pressure components vectors                    #
#       - p is the modulus of the pressure vector                           #
#############################################################################

tic(mpicomm)
rprint(mpicomm, 'Computing pressure forces...')

px, py, pz, p = compute_pressure(mpicomm, universe, F, B)

toc(mpicomm, 'Computing pressure forces')



#############################################################################
# Computation of energy fluxes                                              #
#                                                                           #
# ene = compute_energy(Comm, Universe, F, B)                                #
#   Calculates the net energy fluxes on surfaces.                           #
#       - Comm is an object of the Comm class (MPI communicator info)       #
#       - Universe is an object of the Universe class (mesh and gas info)   #
#       - F is the view factor matrix                                       #
#       - B is the surface flux vector                                      #
#       - ene is the energy flux vector                                     #
#############################################################################

tic(mpicomm)
rprint(mpicomm, 'Computing energy fluxes...')

ene = compute_energy(mpicomm, universe, F, B)

toc(mpicomm, 'Computing energy fluxes')


#############################################################################
# Computation of total force and moment                                     #
#                                                                           #
# tot_force(Comm, Universe, ID, B, px, py, pz)                              #
#   Calculates the total force.                                             #
#       - Comm is an object of the Comm class (MPI communicator info)       #
#       - Universe is an object of the Universe class (mesh and gas info)   #
#       - ID is the int identifier of the group of surfaces to calculate    #
#         the total force on                                                #
#       - B is the surface flux vector                                      #
#       - px, py, pz are the pressure components vectors                    #
#                                                                           #
# tot_moment(Comm, Universe, ID, B, px, py, pz, point)                      #
#   Calculates the total force.                                             #
#       - Comm is an object of the Comm class (MPI communicator info)       #
#       - Universe is an object of the Universe class (mesh and gas info)   #
#       - ID is the int identifier of the group of surfaces to calculate    #
#         the total moment on                                               #
#       - B is the surface flux vector                                      #
#       - px, py, pz are the pressure components vectors                    #
#       - point are the [x,y,z] coordinates of the pivot pointof the moment #
#############################################################################

force = tot_force(mpicomm, universe, 0, B, px, py, pz)
moment = tot_moment(mpicomm, universe, 0, B, px, py, pz, [0.0, 0.0, 0.0])

# Some example calculations for the total force/moment on a surface.

if mpicomm.rank == 0:
    rprint(mpicomm, 'Force is: {force} N'.format(force=force))
    rprint(mpicomm, 'In modulus: {force} N'.format(force=np.sqrt(np.sum(force**2))))

    rprint(mpicomm, 'Moment is: {moment} Nm'.format(moment=moment))
    rprint(mpicomm, 'In modulus: {moment} Nm'.format(moment=np.sqrt(np.sum(moment**2))))
    
    drag = force/(1.38064852e-23*universe.T[1]*universe.S[1]**2*universe.n[1]*A)
    rprint(mpicomm, 'Drag coefficient is: {drag}'.format(drag=drag))


#############################################################################
# File output:                                                              #
#                                                                           #
# - Specify filename (including path)                                       #
# - Specify variables to output and respective labels                       #
#                                                                           #
# write_vtk(Comm, Universe, filename, celldata, labels)                     #
#   Outputs the data in celldata to a VTK file                              #
#       - Comm is an object of the Comm class (MPI communicator info)       #
#       - Universe is an object of the Universe class (mesh and gas info)   #
#       - filename is the filename including the path with extension *.vtp  #
#       - celldata is a list of the data vectors to output                  #
#       - labels is the list of corresponding label names                   #
#############################################################################

if mpicomm.rank == 0:
    filename = 'VTKHemiOut.vtp'
    
    print('Writing result to file: {fname}'.format(fname=filename))

    write_vtk(mpicomm, universe, filename, [B, px, py, pz, p, ene], ['Flux', 'px', 'py', 'pz', 'p', 'E'])
    # write_vtk(filename, [B, px, py, pz, p, F[2428]], ['Flux', 'Px', 'Py', 'Pz', 'P', 'F2428']) # Example, saves the view factors for surface # 2428 for diagnostic purposes
