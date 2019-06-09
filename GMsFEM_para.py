

import scipy.sparse as sparse
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import cg
import matplotlib.pylab as plt	
import numpy.matlib
from mpi4py import MPI
import para_cg 
## partition of the mesh


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank ==0:
	print ""
	print " GMsFEM for elliptic equation on uniform mesh "
	print " Python version "
	print " zero Dirichlet boundary condition "
	print " use mpi when computing the partition of unity and eigenbasis"
	print "\n"
## mesh
Lx=1.;Ly=1.;Nx=10;Ny=Nx;nx=10;ny=nx
#Lx=1.;Ly=1.;Nx=5;Ny=5;nx=200;ny=200
Hx=Lx/Nx;Hy=Ly/Ny;hx=Lx / (nx*Nx);hy=Ly/(ny*Ny)
nnx=nx*Nx+1;nny=ny*Ny+1
n_nodes=(nx*Nx+1)*(ny*Ny+1)
n_elements=nx*ny*Nx*Ny
#print "hx is % 2.2f " %  (hx)
(bdnodes,left_nodes,right_nodes,bottom_nodes,top_nodes,free_nodes)=cg.getboundary_dof(nny-1,nnx-1);n_bdnodes=bdnodes.shape[0]
x = np.linspace(hx/2, Lx-hx/2, nx)
y = np.linspace(hy/2, Ly-hy/2, ny)
xc, yc = np.meshgrid(x, y)
nbasis=10##------------------------------------------------------------------------------------------
if nx==10 and ny==10 and Nx==10 and Ny==10:
 stiff_coeff=cg.loadmatrix('k1.txt')
 stiff_coeff=np.reshape(stiff_coeff,(100,100)).transpose()
else:
 stiff_coeff=np.ones((nny-1,nnx-1))
local_massmatrix=cg.local_mass(hx,hy)
local_stiffmatrix=cg.local_stiff(hx,hy)
Astiff=cg.assembleweightmatrix(stiff_coeff,local_stiffmatrix)
## assemble global matrix and force
if rank ==0:
	print " Assembling fine scale matrix... "
	#Massweight=cg.assembleweightmatrix(stiff_coeff,local_massmatrix)
	Massforce=cg.assembleweightmatrix(np.ones((nny-1,nnx-1)),local_massmatrix)
	f=np.ones(n_nodes);
	F=Massforce.dot(np.transpose(f));#print F


	Astiff_free=Astiff[free_nodes][:,free_nodes];
	F_free=F[free_nodes]
	print " Solving linear system... "
	print "\n"
	u=np.zeros(n_nodes);
	u[free_nodes]=spla.spsolve(Astiff_free, F_free)
	#cg.plot_vector(u,nnx,nny)
start_c,end_c=para_cg.partition(size,rank,Nx*Ny) ## partition the coarse element
comm.Barrier()

## MsFEM --------------------------------------------------------------------------------------------####

### compute basis with mpi
if rank==0:
	print " Computing partition of unity..."
basis_para,I_para=para_cg.get_elementbasis_par(nx,ny,Nx,Ny,Astiff,start_c,end_c)
#if rank==1:
#	print basis_para.shape
'''
basis=comm.gather(basis_para,root=0)
basis = np.asarray(basis)
'''

start_allc = comm.gather(start_c, root=0)
end_allc = comm.gather(end_c, root=0)

if rank==0:
	basis=np.zeros( (4,(nx+1)*(ny+1),Nx*Ny))
	basis[:,:,start_c:end_c+1]= basis_para
	I=np.zeros((Nx*Ny,(nx+1)*(ny+1)),dtype=np.int)
	I[start_c:end_c+1,:]= I_para
## assemble basis
if rank == 0:
        for i in range(1, size):
		recv_buffer=np.zeros( (4,(nx+1)*(ny+1),end_allc[i]-start_allc[i]+1))
                comm.Recv(recv_buffer, source=i,tag=0)
                basis[:,:,start_allc[i]:end_allc[i]+1]= recv_buffer
		recv_buffer0=np.zeros( ( end_allc[i]-start_allc[i]+1,(nx+1)*(ny+1)),dtype=np.int)
                comm.Recv(recv_buffer0, source=i,tag=1)
                I[start_allc[i]:end_allc[i]+1,:]= recv_buffer0
else:
        # all other process send their result
        comm.Send(basis_para,dest=0,tag=0)
	comm.Send(I_para,dest=0,tag=1)

### no parallel for reconstruct and compute ms solution
comm.Barrier()
if rank==0:
	print " Assemble MsFEM basis matrix and solve MsFEM system..."
	phimatrix=cg.restructurebasis(Nx,Ny,nx,ny,I,basis)
	umsfine0=cg.solve_ms(free_nodes,phimatrix,Astiff_free,F_free)
	umsfine0[bdnodes]=0
	print " L2 error of MsFEM solution is % 2.3f " % la.norm(np.subtract(u, umsfine0)/la.norm(u))
	print "\n"
	#cg.plot_vector(umsfine0,nnx,nny)


## GMsFEM --------------------------------------------------------------------------------------------####

### the coefficient for the mass matrix is still kappa
#mass_coeff=stiff_coeff

#if rank==0:
#	mass_coeff=cg.compute_ktilda(hx,hy,stiff_coeff,phimatrix)
#else:	
#	mass_coeff=None
if rank!=0:
	phimatrix=None
phimatrix=comm.bcast(phimatrix,root=0)
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
f_snap=sparse.csr_matrix(f_snap)
if rank==0:
	print " computing  eigenbasis... "

start_n,end_n=para_cg.partition(size,rank,(Nx+1)*(Ny+1)) ## partition the coarse nodes
start_alln = comm.gather(start_n, root=0)
end_alln = comm.gather(end_n, root=0)
for node in range(start_n,end_n+1):
  i,j=para_cg.index1dto2d(node+1,Nx+1,Ny+1)
  	

  if rank==0 and (end_n+1-node)%Nx==0:
   print " rest coarse neighborhood is ",end_n+1-node
  if i==1 or i==Nx+1 or j==1 or j==Ny+1:
   localphi=phimatrix[node,:]
   nonzero=np.nonzero(localphi);nonzero=nonzero[1]
   iy=np.append(iy, nonzero);value=phimatrix[node,nonzero]
   ivalue=np.append(ivalue,value.toarray())

  else:
   gi=cg.getCoarseNeighborNodes(i,j,Nx,Ny,nx,ny)
# assemble local matrix and find eigenvalue and eigenvector

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

   d1,v1 = sparse.linalg.eigsh(A_eig,k=nx*2,M=M_eig,which='SM')
   v=usnap.dot(v1)
   v=v[:,0:nbasis];v[:,0]=1

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
## assemble basis 
   local_partition=phimatrix[node,gi].toarray()
   local_partition=np.tile(np.transpose(local_partition), (1, nbasis))
   nodebasis=v*local_partition
   tempvalue=np.reshape(np.transpose(nodebasis),nbasis*(2*nx+1)*(2*ny+1))

   tempy=np.tile(gi,(nbasis,1))
   tempy=np.reshape(tempy,nbasis*(2*nx+1)*(2*ny+1))

 
   iy=np.append(iy, tempy)
   ivalue=np.append(ivalue,tempvalue)

###### gather iy,ivalue
if rank==0:
	ixall=para_cg.getix(Nx,Ny,nx,ny,phimatrix,nbasis)

comm.Barrier()
iylengthtotal = np.zeros((1),dtype='int')
iylength = np.zeros((1),dtype='int')
iylength[0]=iy.shape[0]

comm.Reduce(iylength, iylengthtotal, op=MPI.SUM, root=0)
#
iylength = comm.gather(iy.shape[0], root=0)


if rank==0:
	
	iyall=np.zeros((iylengthtotal),dtype='int')
	ivalueall=np.zeros((iylengthtotal))
	
	iyall[0:iy.shape[0]]=iy
	ivalueall[0:iy.shape[0]]=ivalue

#comm.Barrier()	
## assemble basis
if rank == 0:
        for i in range(1, size):
		
		recv_buffer1=np.zeros((iylength[i]),dtype='int')
		recv_buffer2=np.zeros((iylength[i]))

		comm.Recv(recv_buffer1, source=i,tag=1)
		comm.Recv(recv_buffer2, source=i,tag=2)

		iyall[sum(iylength[0:i]):sum(iylength[0:i+1])]=recv_buffer1
		ivalueall[sum(iylength[0:i]):sum(iylength[0:i+1])]=recv_buffer2
else:
        # all other process send their result

	comm.Send(iy,dest=0,tag=1)
	comm.Send(ivalue,dest=0,tag=2)


if rank==0:
	print " Assemble GMsFEM basis matrix and solve GMsFEM system..."
	phimatrix_enrich=sparse.csr_matrix((ivalueall, (ixall, iyall)), shape=(2*Nx+2*Ny+nbasis*(Nx-1)*(Ny-1),nnx*nny))
	#### solve GMsFEM solution
	umsfine=cg.solve_ms(free_nodes,phimatrix_enrich,Astiff_free,F_free)
	umsfine[bdnodes]=0
	print " L2 error of GMsFEM solution is % 2.3f " % la.norm(np.subtract(u, umsfine)/la.norm(u))

	#cg.plot_vector(umsfine,nnx,nny)
