import scipy.sparse as sparse
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import cg
import matplotlib.pylab as plt	
import numpy.matlib

plt.close('all')
print ""
print " FEM for elliptic equation on uniform mesh "
print " Python2 version "
print " zero Dirichlet boundary condition \n"
plt.close('all')
## mesh
Lx=1.;Ly=1.;Nx=1;Ny=1;nx=100;ny=100
#Lx=1.;Ly=1.;Nx=1;Ny=1;nx=3;ny=2
Hx=Lx/Nx;Hy=Ly/Ny;hx=Lx / (nx*Nx);hy=Ly/(ny*Ny)
nnx=nx*Nx+1;nny=ny*Ny+1
n_nodes=(nx*Nx+1)*(ny*Ny+1)
n_elements=nx*ny*Nx*Ny
#print "hx is % 2.2f " %  (hx)
(bdnodes,left_nodes,right_nodes,bottom_nodes,top_nodes,free_nodes)=cg.getboundary_dof(nnx-1,nny-1);n_bdnodes=bdnodes.shape[0]
x = np.linspace(hx/2, Lx-hx/2, nx)
y = np.linspace(hy/2, Ly-hy/2, ny)
xc, yc = np.meshgrid(x, y)

## model
'''
high_value=np.power(10,6)
stiff_coeff=np.ones((nny-1,nnx-1))
stiff_coeff[ ny+1:ny*2,nx+1:(Nx-2)*nx]=high_value
stiff_coeff[ ny*3:ny*4-2,nx+1:(Nx-2)*nx]=high_value
stiff_coeff[ ny*5:ny*6-2,nx+5:(Nx-3)*nx]=high_value
stiff_coeff[ ny*7:ny*8-2,nx+5:(Nx-3)*nx]=high_value
'''
stiff_coeff=np.ones((nny-1,nnx-1))
#stiff_coeff=cg.loadmatrix('k1.txt')
#stiff_coeff=np.reshape(stiff_coeff,(100,100)).transpose()
#print stiff_coeff
#cg.plot_matrix(stiff_coeff,Lx,Ly)
#cg.assembleweightstiff(hx,hy,stiff_coeff)
#ss=np.linspace(1, 5,5)

## assemble global matrix and force
print " Assembling fine scale matrix... "
local_massmatrix=cg.local_mass(hx,hy)
local_stiffmatrix=cg.local_stiff(hx,hy)
Astiff=cg.assembleweightmatrix(stiff_coeff,local_stiffmatrix)
#Massweight=cg.assembleweightmatrix(stiff_coeff,local_massmatrix)
Massforce=cg.assembleweightmatrix(np.ones((nny-1,nnx-1)),local_massmatrix)
f=np.ones((n_nodes,1));
F=Massforce.dot(f);#print F

#print "Setting Dirichlet boundary condition..."
#F[bdnodes]=0;#print F
#Astiff=sparse.lil_matrix(Astiff)
#Astiff[bdnodes,:] = sparse.lil_matrix((n_bdnodes, n_nodes))
#for i in range(0, n_bdnodes):
#	Astiff[bdnodes[i],bdnodes[i]]=1

Astiff_free=Astiff[free_nodes][:,free_nodes];
F_free=F[free_nodes]
print " Solving linear system... "
#u=spla.spsolve(Astiff, F)
#u=spla.spsolve(Astiff, F,  use_umfpack=True)
#u=np.reshape(u,(nny,nnx))
u=np.zeros(n_nodes);
u[free_nodes]=spla.spsolve(Astiff_free, F_free)
#u[free_nodes]=spla.spsolve(Astiff_free, F_free,  use_umfpack=False)
#um=np.reshape(u,(nny,nnx))
#cg.plot_matrix(um,Lx,Ly)
cg.plot_vector(u,nnx,nny)
#plt.close("all")
