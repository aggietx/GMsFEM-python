import scipy.sparse as sparse
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import cg
import matplotlib.pylab as plt	
import numpy.matlib


plt.close('all')
print ""
print " GMsFEM for elliptic equation on uniform mesh "
print " Python version "
print " zero Dirichlet boundary condition \n"
plt.close('all')
## mesh
#Lx=1.;Ly=1.;Nx=5;Ny=5;nx=5;ny=5
Lx=1.;Ly=1.;Nx=10;Ny=10;nx=10;ny=10
Hx=Lx/Nx;Hy=Ly/Ny;hx=Lx / (nx*Nx);hy=Ly/(ny*Ny)
nnx=nx*Nx+1;nny=ny*Ny+1
n_nodes=(nx*Nx+1)*(ny*Ny+1)
n_elements=nx*ny*Nx*Ny
#print "hx is % 2.2f " %  (hx)
(bdnodes,left_nodes,right_nodes,bottom_nodes,top_nodes,free_nodes)=cg.getboundary_dof(nnx-1,nny-1);n_bdnodes=bdnodes.shape[0]
x = np.linspace(hx/2, Lx-hx/2, nx)
y = np.linspace(hy/2, Ly-hy/2, ny)
xc, yc = np.meshgrid(x, y)
### number of basis used
nbasis=10 ### less than 2*nx###########################################################################

## model
'''
high_value=np.power(10,6)
stiff_coeff=np.ones((nny-1,nnx-1))
stiff_coeff[ ny+1:ny*2,nx+1:(Nx-2)*nx]=high_value
stiff_coeff[ ny*3:ny*4-2,nx+1:(Nx-2)*nx]=high_value
stiff_coeff[ ny*5:ny*6-2,nx+5:(Nx-3)*nx]=high_value
stiff_coeff[ ny*7:ny*8-2,nx+5:(Nx-3)*nx]=high_value
'''
stiff_coeff=cg.loadmatrix('k1.txt')
stiff_coeff=np.reshape(stiff_coeff,(100,100)).transpose()
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
f=np.ones(n_nodes);
F=Massforce.dot(np.transpose(f));#print F

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
############### Compute partition of unity  #######
print " Compute partition of unity... "
phimatrix=cg.get_ms_pu(nx,ny,Nx,Ny,Astiff)

umsfine0=cg.solve_ms(free_nodes,phimatrix,Astiff_free,F_free)
umsfine0[bdnodes]=0
print " L2 error of MsFEM is % 2.3f " % la.norm(np.subtract(u, umsfine0)/la.norm(u))
cg.plot_vector(umsfine0,nnx,nny)

## GMsFEM 
### the coefficient for the mass matrix is still kappa
#mass_coeff=stiff_coeff
mass_coeff=cg.compute_ktilda(hx,hy,stiff_coeff,phimatrix)
ix=np.empty([0, 0], dtype=int)
iy=np.empty([0, 0], dtype=int)
ivalue=np.empty([0, 0])
gind=0


nei_bdinfor=cg.getboundary_dof(2*ny,2*nx);nei_bdnodes=nei_bdinfor[0];nei_freenodes=nei_bdinfor[5]
regular=np.power(10, -10.)*np.true_divide(np.amin(stiff_coeff),np.amax(stiff_coeff))
f_snap = sparse.lil_matrix(((2*nx+1)*(2*ny+1), 4*nx+4*ny))

for i in range(0, 4*nx+4*ny):
	f_snap[nei_bdnodes[i],i]=1
f_snap=sparse.csc_matrix(f_snap)

print " computing  eigenbasis... "
for j in range(1, Ny+2):
 for i in range(1, Nx+2):
  node=i+(j-1)*(Nx+1)-1
  if ((Nx+1)*(Ny+1)-node)%Nx==0:
   print "rest coarse neighborhood is ",(Nx+1)*(Ny+1)-node
  if i==1 or i==Nx+1 or j==1 or j==Ny+1:
   localphi=phimatrix[node,:]
   nonzero=np.nonzero(localphi);nonzero=nonzero[1]
   ix=np.append(ix, gind*np.ones((nonzero.shape), dtype=int) )
   iy=np.append(iy, nonzero);value=phimatrix[node,nonzero]
   ivalue=np.append(ivalue,value.toarray())
   gind=gind+1
  else:
   gi=cg.getCoarseNeighborNodes(i,j,Nx,Ny,nx,ny)
# assemble local matrix and find eigenvalue and eigenvector
   #ge=getCoarseNeighborElems(i,j,Nx,Ny,nx,ny)
   localstiff_coeff=stiff_coeff[(j-2)*ny:j*ny,(i-2)*nx:i*nx]
   localmass_coeff=mass_coeff[(j-2)*ny:j*ny,(i-2)*nx:i*nx]
   local_Astiff=cg.assembleweightmatrix(localstiff_coeff,local_stiffmatrix)
   local_Mass=cg.assembleweightmatrix(localmass_coeff,local_massmatrix)
   ## harmonic snapshot############################################################## more stable to compute eigenvalue and eigenvector
   
   localF= -local_Astiff.dot(f_snap);localFree=localF[nei_freenodes,:];localFree=localFree.toarray()

   localAfree=local_Astiff[nei_freenodes][:,nei_freenodes];
   usnap=np.zeros(( (2*nx+1)*(2*ny+1),4*nx+4*ny ))
   usnap[nei_freenodes,:]=spla.spsolve(localAfree, localFree, use_umfpack=True)
   usnap[nei_bdnodes,:]=f_snap[nei_bdnodes,:].toarray()
   usnap=sparse.csr_matrix(usnap)
   A_eig=np.transpose(usnap)*local_Astiff*usnap;A_eig=A_eig+A_eig.transpose()
   M_eig=np.transpose(usnap)*local_Mass*usnap;M_eig=M_eig+M_eig.transpose()
   A_eig=cg.regularization(regular,A_eig)
   M_eig=cg.regularization(regular,M_eig)
#   '''
   d1,v1 = sparse.linalg.eigsh(A_eig,k=nx*2,M=M_eig,which='SM')
   v=usnap.dot(v1)
   v=v[:,0:nbasis];v[:,0]=1
#   '''
   '''
   ## finescale snapshot###############################################################
   local_Astiff = (local_Astiff + local_Astiff.transpose())
   local_Mass = (local_Mass + local_Mass.transpose())
## regularization 
   local_Astiff=cg.regularization(regular,local_Astiff)
   local_Mass=cg.regularization(regular,local_Mass)
## solve eigenvalue problem

   d,v = sparse.linalg.eigsh(local_Astiff,k=nx*2,M=local_Mass,which='SM')
   v[:,0]=1
   v=v[:,0:nbasis]
   '''
## assemble
   local_partition=phimatrix[node,gi].toarray()
   local_partition=np.tile(np.transpose(local_partition), (1, nbasis))
   nodebasis=v*local_partition
   tempvalue=np.reshape(np.transpose(nodebasis),nbasis*(2*nx+1)*(2*ny+1))
   tempx=np.tile(range(gind, gind+nbasis),((2*nx+1)*(2*ny+1),1))
   tempx=np.reshape(np.transpose(tempx),nbasis*(2*nx+1)*(2*ny+1))
   tempy=np.tile(gi,(nbasis,1))
   tempy=np.reshape(tempy,nbasis*(2*nx+1)*(2*ny+1))

   ix=np.append(ix, tempx)
   iy=np.append(iy, tempy)
   ivalue=np.append(ivalue,tempvalue)

   #print gind
   gind=gind+nbasis

phimatrix_enrich=sparse.csr_matrix((ivalue, (ix, iy)), shape=(2*Nx+2*Ny+nbasis*(Nx-1)*(Ny-1),nnx*nny))

#### solve GMsFEM solution
umsfine=cg.solve_ms(free_nodes,phimatrix_enrich,Astiff_free,F_free)
umsfine[bdnodes]=0
print " L2 error of GMsFEM is % 2.3f " % la.norm(np.subtract(u, umsfine)/la.norm(u))
cg.plot_vector(umsfine,nnx,nny)




