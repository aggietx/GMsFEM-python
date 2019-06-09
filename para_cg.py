import scipy.sparse as sparse
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pylab as plt	
import numpy.matlib
import operator
from os import listdir
import cg
def index1dto2d(i,nx,ny):

	if i%nx==0:
	 x=nx
	 y=i/nx
	else:
	 x=i%nx
	 y=(i-x)/nx+1
	return x, y

def get_elementbasis_par(nx,ny,Nx,Ny,A,start_c,end_c):
#### get multiscale partition of unity(mpi version)  
##### A is the global stiffness matrix
	(local_bddof,local_left_nodes,local_right_nodes,local_bottom_nodes,local_top_nodes,free_nodes)=cg.getboundary_dof(nx,ny)
	n_coarseelement=end_c-start_c+1
	msbasis=np.zeros( (4,(nx+1)*(ny+1),n_coarseelement))
	I=np.zeros((n_coarseelement,(nx+1)*(ny+1)),dtype=np.int)
	cind=0
	#for j in range(1, Ny+1):
		#for i in range(1, Nx+1):	
	for ne in range(start_c,end_c+1):	
			i,j=index1dto2d(ne+1,Nx,Ny)

			gi=cg.getCoarsenodeselementinside(i,j,Nx,Ny,nx,ny)
			#print gi
			localA=A[gi,:];localA=localA[:,gi]
			msbasis[:,:,cind]=cg.compute_msbasis(nx,ny,localA,local_bddof,local_left_nodes,local_right_nodes,local_bottom_nodes,local_top_nodes,free_nodes)
			I[cind,:]=gi
			cind=cind+1
			#print cind
	
	return msbasis,I

def partition(size,rank,M):
	start = rank*(M/size) 
	if (M%size) < rank:
	 start +=M%size
	else:
	 start +=rank
	end   = start + M/size + ((M%size) > rank)-1
	return start,end
def getix(Nx,Ny,nx,ny,phimatrix,nbasis):
	ix=np.empty([0, 0], dtype=int)
	gind=0
	for j in range(1, Ny+2):
 	 for i in range(1, Nx+2):
          node=i+(j-1)*(Nx+1)-1
	  if i==1 or i==Nx+1 or j==1 or j==Ny+1:
           localphi=phimatrix[node,:]
           nonzero=np.nonzero(localphi);nonzero=nonzero[1]
           ix=np.append(ix, gind*np.ones((nonzero.shape), dtype=int) )
   
           gind=gind+1
          else:
	   tempx=np.tile(range(gind, gind+nbasis),((2*nx+1)*(2*ny+1),1))
           tempx=np.reshape(np.transpose(tempx),nbasis*(2*nx+1)*(2*ny+1))
	   ix=np.append(ix, tempx)
	   gind=gind+nbasis
	return ix
