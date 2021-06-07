import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import transpose

# system dynamics
A=np.array([[0.9974,0.0539],[-0.1078,1.1591]])
B=np.array([[0.0013],[0.0539]])

# Cost function definition
H=np.array([[2,0],[0,0]])
Q=np.array([[2,0],[0,1]])
R=2

N=500
F=np.zeros((1,2))
P=H

# initial values
x0=np.array([[1],[-1]])
x=x0
u=0
v=[]
y1=[]
y2=[]

# DLQR algorithm
for i in range(N):
    F=-np.linalg.inv(R+(B.transpose()).dot(P.dot(B))).dot(B.transpose()).dot(P.dot(A))
    y1.append(x[0,0])
    y2.append(x[1,0])
    u=F.dot(x)
    x=A.dot(x)+B*(u)
    P=((A+B.dot(F)).transpose()).dot(P.dot((A+B.dot(F))))+R*(F.transpose()).dot(F)+Q
    v.append(u.tolist())

# flattening the input u
from itertools import chain
v = list(chain.from_iterable(v))
print(v)

# time steps
n=[]
for i in range(N):
    n.append(i)
    i=i+1

# plotting the figures
fig, axs = plt.subplots(2)

axs[0].plot(n,y1, label="X1")
axs[0].plot(n,y2, label="X2")
axs[0].set_title("State Variables")
axs[0].set( ylabel="X1, X2")
axs[0].legend()
axs[1].plot(n,v, label="U")
axs[1].set_title("Control Signal")
axs[1].set(xlabel="N (time step)", ylabel="U")

plt.show()

J_opt=0.5*x0.transpose().dot(P.dot(x0))
print(J_opt)