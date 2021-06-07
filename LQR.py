import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import reshape, transpose

# System dynamics

B=np.array([[0],[1]])

# Cost function definition

H=np.array([[1,0],[0,0]])
Q=np.array([[1,0],[0,1]])
R=10

# Time

ts=0.001
tf=10
t=np.arange(0,tf,ts)
time=np.array(t)
N=np.count_nonzero(time>=0)

# K calculation

A=np.zeros((2,2))
k=np.zeros((N,2,2))
k[N-1,:,:]=H

for i in range(N-2,0,-1):
    A = np.array([[0,2-np.cos(time[i])],[-1,-2+np.sin(time[i])]])
    k[i-1,:,:] = k[i,:,:] + ts*( Q -k[i,:,:].dot((R**-1)*B.dot(B.transpose().dot(k[i,:,:])))+k[i,:,:].dot(A)+A.dot(k[i,:,:] ))

K11 = k[:,0,0]
K12 = k[:,0,1]
K22 = k[:,1,1]

fig, axs = plt.subplots(3)

axs[0].plot(time,K11, label="K11")
axs[1].plot(time,K12, label="K12")
axs[2].plot(time,K22, label="K22")

axs[0].legend()
axs[1].legend()
axs[2].legend()

# Optimal Control Signal

x = np.zeros((N,2))  
x[0,:] = np.array([10,0])
u = np.zeros((1,N)) 

for i in range(0,N-1,1):
    
    u[0,i] = -R**(-1)*B.transpose().dot(k[i,:,:].dot(x[i,:].transpose()))
    A = np.array([[0,2-np.cos(time[i])],[-1,-2+np.sin(time[i])]])
    x[i+1,:] =x[i,:]+ts*(A.dot(x[i,:].transpose()) + B.dot(u[:,i]))

# Plotting The Results

x1=x[:,0]
x2=x[:,1]
U=u[0,:]
fig, axs = plt.subplots(2)

axs[0].plot(time,x1, label="x1")
axs[0].plot(time,x2, label="x2")
axs[1].plot(time,U, label="U")

axs[0].set_title("State Variables")
axs[1].set_title("Control Signal")

axs[0].legend()
axs[1].set(xlabel="time", ylabel="U")
plt.show()

# Optimal Cost function
J_opt = 0.5*x1[-1]**2 + ts*np.sum(x1.dot(x1.transpose()) + x2.dot(x2.transpose()) + R*u.dot(u.transpose()))
print(J_opt)