import scipy.sparse as sparse
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import matplotlib.pylab as plt	
import numpy.matlib
import operator
from os import listdir
def local_stiff(hx,hy):
	local_matrix=np.zeros((4,4))
	local_matrix[0,0]=(hx*hx+hy*hy)/(3*hx*hy)
	local_matrix[0,1]=hx/(6*hy)-hy/(3*hx)
	local_matrix[0,2]=-(hx*hx+hy*hy)/(6*hx*hy)
	local_matrix[0,3]=-hx/(3*hy)+hy/(6*hx)

	local_matrix[1,0]=local_matrix[0,1]
	local_matrix[1,1]=local_matrix[0,0]
	local_matrix[1,2]=local_matrix[0,3]
	local_matrix[1,3]=local_matrix[0,2]

	local_matrix[2,0]=local_matrix[0,2]
	local_matrix[2,1]=local_matrix[1,2]
	local_matrix[2,2]=local_matrix[0,0]
	local_matrix[2,3]=local_matrix[0,1]

	local_matrix[3,0]=local_matrix[0,3]
	local_matrix[3,1]=local_matrix[1,3]
	local_matrix[3,2]=local_matrix[2,3]
	local_matrix[3,3]=local_matrix[0,0]
	return local_matrix

def local_mass(hx,hy):
	local_matrix=np.zeros((4,4))
	local_matrix[0,0]=1./9
	local_matrix[0,1]=1./18
	local_matrix[0,2]=1./36
	local_matrix[0,3]=1./18

	local_matrix[1,0]=local_matrix[0,1]
	local_matrix[1,1]=local_matrix[0,0]
	local_matrix[1,2]=local_matrix[0,3]
	local_matrix[1,3]=local_matrix[0,2]

	local_matrix[2,0]=local_matrix[0,2]
	local_matrix[2,1]=local_matrix[1,2]
	local_matrix[2,2]=local_matrix[0,0]
	local_matrix[2,3]=local_matrix[0,1]

	local_matrix[3,0]=local_matrix[0,3]
	local_matrix[3,1]=local_matrix[1,3]
	local_matrix[3,2]=local_matrix[2,3]
	local_matrix[3,3]=local_matrix[0,0]
	local_matrix=hx*hy*local_matrix
	return local_matrix
	
def localdof_2globaldof(ny,nx):
	ngrid=nx*ny

	total_nodes=(nx+1)*(ny+1)
	all_nodes_vec=np.arange(0,total_nodes)
	all_nodes=np.reshape(all_nodes_vec,(ny+1,nx+1))
	nod=np.zeros((4,ngrid),dtype=np.int)
	sx=all_nodes[0:ny,0:nx];first_node=np.reshape(sx,nx*ny);
	nod[0,:]=first_node
	nod[1,:]=first_node+1
	nod[2,:]=first_node+nx+2
	nod[3,:]=first_node+nx+1
	
	gridx=np.matlib.repmat(nod, 4, 1)
	gridz=np.zeros((16,ngrid),dtype=np.int)
	aa=np.matlib.repmat(first_node, 4, 1)
	gridz[0:4,:]=np.matlib.repmat(first_node, 4, 1)
	gridz[4:8,:]=np.matlib.repmat(first_node+1, 4, 1)
	gridz[8:12,:]=np.matlib.repmat(first_node+nx+2, 4, 1)
	gridz[12:16,:]=np.matlib.repmat(first_node+nx+1, 4, 1)
	return (gridx,gridz)

def assembleweightmatrix(coeff,local_matrix):
	nx=coeff.shape[1]
	ny=coeff.shape[0]
	ngrid=nx*ny
	(gridx,gridz)=localdof_2globaldof(ny,nx)
	total_nodes=(nx+1)*(ny+1)

	coeff_vec=np.reshape(coeff,nx*ny)
	coeffvec=np.reshape(coeff,(1,nx*ny));#print k
	lk=np.reshape(local_matrix,(16,1))
	tempk=lk.dot(coeffvec)

	gdx=np.reshape(gridx,ngrid*16)
	gdy=np.reshape(gridz,ngrid*16)
	gdvalue=np.reshape(tempk,ngrid*16)

	
	globalmatrix=sparse.csr_matrix((gdvalue, (gdy, gdx)), shape=(total_nodes,total_nodes))
	#globalmatrix=sparse.lil_matrix(globalmatrix)
	return globalmatrix


def getboundary_dof(nx,ny):
	total_nodes=(nx+1)*(ny+1)
	all_nodes_vec=np.arange(0,total_nodes)
	all_nodes=np.reshape(all_nodes_vec,(ny+1,nx+1))
	left_nodes=all_nodes[:,0]
	right_nodes=all_nodes[:,nx]
	bottom_nodes=all_nodes[0,:]
	top_nodes=all_nodes[ny,:]
	free_nodes=all_nodes[1:ny,1:nx];free_nodes=np.reshape(free_nodes,(ny-1,nx-1))
	free_nodes=np.reshape(free_nodes,(nx-1)*(ny-1));
	boundary_nodes=np.setdiff1d(all_nodes_vec,free_nodes)
	return (boundary_nodes,left_nodes,right_nodes,bottom_nodes,top_nodes,free_nodes)
def getcoarseelementinside(i,j,Nx,Ny,nx,ny):
### global elements array inside coarse element (j,i)
	total_element=nx*ny*Nx*Ny
	all_element_vec=np.arange(0,total_element)
	all_element=np.reshape(all_element_vec,(ny*Ny,nx*Nx))
	gi=all_element[(j-1)*ny:j*ny,(i-1)*nx:i*nx]
	gi=np.reshape(gi,nx*ny)
	return gi
def getCoarseNeighborElems(i,j,Nx,Ny,nx,ny):
### global elements array inside coarse neighborhood (j,i)
	total_element=nx*ny*Nx*Ny
	all_element_vec=np.arange(0,total_element)
	all_element=np.reshape(all_element_vec,(ny*Ny,nx*Nx))
	gi=all_element[(j-2)*ny:j*ny,(i-2)*nx:i*nx]
	gi=np.reshape(gi,4*nx*ny)
	return gi
def getCoarseNeighborNodes(i,j,Nx,Ny,nx,ny):
### global nodes array inside coarse neighborhood (j,i)
	
	total_nodes=(Ny*ny+1)*(Nx*nx+1)
	all_nodes_vec=np.arange(0,total_nodes)
	all_nodes=np.reshape(all_nodes_vec,(ny*Ny+1,nx*Nx+1))
	gi=all_nodes[(j-2)*ny:j*ny+1,(i-2)*nx:i*nx+1]
	gi=np.reshape(gi,(2*nx+1)*(2*ny+1))
	return gi
def getCoarsenodeselementinside(i,j,Nx,Ny,nx,ny):
### global ndoes array inside coarse element (j,i)
	total_nodes=(Ny*ny+1)*(Nx*nx+1)
	all_nodes_vec=np.arange(0,total_nodes)
	all_nodes=np.reshape(all_nodes_vec,(ny*Ny+1,nx*Nx+1))
	gi=all_nodes[(j-1)*ny:j*ny+1,(i-1)*nx:i*nx+1]
	gi=np.reshape(gi,(nx+1)*(ny+1))
	return gi
def get_ms_pu(nx,ny,Nx,Ny,A):
#### get multiscale partition of unity 
##### A is the global stiffness matrix
	(local_bddof,local_left_nodes,local_right_nodes,local_bottom_nodes,local_top_nodes,free_nodes)=getboundary_dof(nx,ny)
	msbasis=np.zeros( (4,(nx+1)*(ny+1),Nx*Ny))
	I=np.zeros((Nx*Ny,(nx+1)*(ny+1)),dtype=np.int)
	cind=0
	for j in range(1, Ny+1):
		for i in range(1, Nx+1):		
			gi=getCoarsenodeselementinside(i,j,Nx,Ny,nx,ny)
			localA=A[gi,:];localA=localA[:,gi]
			msbasis[:,:,cind]=compute_msbasis(nx,ny,localA,local_bddof,local_left_nodes,local_right_nodes,local_bottom_nodes,local_top_nodes,free_nodes)
			I[cind,:]=gi
			cind=cind+1
			#print cind
	phimatrix=restructurebasis(Nx,Ny,nx,ny,I,msbasis)
	return phimatrix


def compute_msbasis(nx,ny,localA,local_bddof,local_left_nodes,local_right_nodes,local_bottom_nodes,local_top_nodes,free_nodes):
	
	F=np.zeros(( (nx+1)*(ny+1),4))
## lower left corner is 1
	F[local_bottom_nodes,0]=np.linspace(1.0, 0.0, num=nx+1)
	F[local_left_nodes,0]=np.linspace(1.0, 0.0, num=ny+1)
## lower right corner is 1
	F[local_bottom_nodes,1]=np.linspace(0.0, 1.0, num=nx+1)
	F[local_right_nodes,1]=np.linspace(1.0, 0.0, num=ny+1)
## upper right corner is 1
	F[local_right_nodes,2]=np.linspace(0.0, 1.0, num=ny+1)
	F[local_top_nodes,2]=np.linspace(0.0, 1.0, num=nx+1)
## upper left corner is 1
	F[local_left_nodes,3]=np.linspace(0.0, 1.0, num=ny+1)
	F[local_top_nodes,3]=np.linspace(1.0, 0.0, num=nx+1)
	F1=-localA.dot(F)
	msbasis=np.zeros(( (nx+1)*(ny+1),4))
	localA_free=localA[free_nodes,:]
	localA_free=localA_free[:,free_nodes]
	msbasis[free_nodes,:]=spla.spsolve(localA_free, F1[free_nodes], use_umfpack=True)
	msbasis[local_bddof,:]=F[local_bddof,:]
	msbasis=msbasis.transpose()
	return msbasis

def restructurebasis(Nx,Ny,nx,ny,I,basis):	
	Ncoarsenode=(Nx+1)*(Ny+1)
	Nglobnode=(Nx*nx+1)*(Ny*ny+1)
	phimatrix=sparse.lil_matrix((Ncoarsenode, Nglobnode))
	nind=0
	for j in range(1, Ny+2):
		for i in range(1, Nx+2):	
			elt=getCoarseNodeNeighbors(i,j,Nx,Ny)
			for kk in range(0,4):
				if kk==0 and elt[kk]>0:
					phimatrix[nind,I[elt[kk]-1,:]]=basis[2,:,elt[kk]-1]
				elif kk==1 and elt[kk]>0:
					phimatrix[nind,I[elt[kk]-1,:]]=basis[3,:,elt[kk]-1]
				elif kk==2 and elt[kk]>0:
					phimatrix[nind,I[elt[kk]-1,:]]=basis[0,:,elt[kk]-1]
				elif kk==3 and elt[kk]>0:
					phimatrix[nind,I[elt[kk]-1,:]]=basis[1,:,elt[kk]-1]
			nind=nind+1
			#print nind
	phimatrix=sparse.csr_matrix(phimatrix)		
	return phimatrix
def getCoarseNodeNeighbors(i,j,Nx,Ny):
# jth node in the y direction
# the output the four coarselements around related to the (i,j)th nodes

	nind=i+(j-1)*(Nx+1)
	elt=np.zeros(4,dtype=np.int)
	elt[0] = (i-1) + (j-2)*Nx;
	elt[1] = elt[0] + 1;
	elt[2] = elt[0] + Nx + 1;
	elt[3] = elt[2] - 1;
	##---Make "null" indices for boundaries

#--left boundary
	if i==1:
  		elt[0] = -1;
  		elt[3] = -1;
	

#---right boundary
	if i==Nx+1:
  		elt[1] = -1;
  		elt[2] = -1; 


#---bottom boundary
	if j==1:
  		elt[0] = -1;
  		elt[1] = -1;


#---top boundary
	if j==Ny+1:
		elt[2] = -1;
		elt[3] = -1;
	return	elt
def solve_ms(free_nodes,phimatrix,Astiff_free,F_free):
	phimatrix_free=phimatrix[:,free_nodes]
	Ams=phimatrix_free*Astiff_free*phimatrix_free.transpose()
	Fms=phimatrix_free.dot(F_free)
	ums=spla.spsolve(Ams, Fms)
	umsfine=phimatrix.transpose().dot(ums)
	return umsfine
def plot_matrix(matrix,Lx,Ly):

	fig = plt.figure()	
	ax =plt.subplots
	plt.imshow(matrix, extent=[0,Lx,0,Ly])
	plt.colorbar()
	plt.draw()
	plt.show(block=False)
def plot_vector(u,nx,ny):
## assume size of u is nx*ny
	u=np.reshape(u,(ny,nx))
	plot_matrix(u,1,1)
def regularization(regular,A):
    A=A+regular*sparse.dia_matrix((A.diagonal(), 0),shape=(A.shape[0], A.shape[1]))
    return A
def getdxdy(nx,ny,hx,hy):
	iy=np.empty([0, 0], dtype=int);ivaluex=np.empty([0, 0]);ivaluey=np.empty([0, 0])
    	total_nodes=(nx+1)*(ny+1)
	all_nodes_vec=np.arange(0,total_nodes)
	all_nodes=np.reshape(all_nodes_vec,(ny+1,nx+1))
	left_bottom=all_nodes[0:ny,:];left_bottom=left_bottom[:,0:nx];
     	left_top=all_nodes[1:ny+1,:];left_top=left_top[:,0:nx];
	right_bottom=all_nodes[0:ny,:];right_bottom=right_bottom[:,1:nx+1];
	right_top=all_nodes[1:1+ny,:];right_top=right_top[:,1:nx+1];
	ix=np.tile(np.linspace(0, nx*ny-1,nx*ny,dtype='int'),4);
	iy=np.append(iy,np.reshape(left_bottom,nx*ny))
	iy=np.append(iy,np.reshape(left_top,nx*ny))
	iy=np.append(iy,np.reshape(right_bottom,nx*ny))
	iy=np.append(iy,np.reshape(right_top,nx*ny))
	
	ivaluex=np.append(ivaluex,-1.*np.ones(nx*ny))
	ivaluex=np.append(ivaluex,-1.*np.ones(nx*ny))
	ivaluex=np.append(ivaluex,1.*np.ones(nx*ny))
	ivaluex=np.append(ivaluex,1.*np.ones(nx*ny))

	ivaluey=np.append(ivaluey,-1.*np.ones(nx*ny))
	ivaluey=np.append(ivaluey,1.*np.ones(nx*ny))
	ivaluey=np.append(ivaluey,-1.*np.ones(nx*ny))
	ivaluey=np.append(ivaluey,1.*np.ones(nx*ny))
	
	Dx=sparse.csr_matrix((ivaluex, (ix, iy)))
	Dy=sparse.csr_matrix((ivaluey, (ix, iy)))
	Dx=.5/hx*Dx;Dy=.5/hy*Dy;
	return(Dx,Dy)
def  compute_ktilda(hx,hy,k,phimatrix):
	nx=k.shape[1];ny=k.shape[0];
	k=np.reshape(k,nx*ny)
	(dx,dy)=getdxdy(nx,ny,hx,hy)
	gradient_sum= (dx*phimatrix.transpose()).power(2)+ ( dy*phimatrix.transpose()).power(2)
	gradient_sum=gradient_sum.sum(1)
	ktilda=np.multiply(gradient_sum.getA1(),k)
	ktilda=np.reshape(ktilda,(nx,ny))
	return ktilda
def loadmatrix(filename):
### read txtfile
  	with open(filename) as file:
         a = [[float(digit) for digit in line.split()] for line in file]

        matrix= np.asarray(a)
	return matrix

