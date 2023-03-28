#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

# Constants
g = 9.81    # acceleration due to gravity
N = 1000   # Number of iterations
t = 0   # initial time
tf = 10 # seconds
dt = (tf-t)/N   # calculate time-step

# Simulation Method
def Sim(m, k, l):
    """
    This definition will use the Euler Cromer method to simulate
    a spring-mass pendulum with the inputs below:

    m: mass of the pendulum
    k: spring constant of the spring-mass pendulum system
    """
    # Calculate L
    L = l + k/(9.8*m)

    # create storage arrays
    px = np.zeros([N])
    py = np.zeros([N])
    vx = np.zeros([N])
    vy = np.zeros([N])
    allx = np.zeros([N])
    ally = np.zeros([N])
    KE = np.zeros([N])
    Us = np.zeros([N])
    Uep = np.zeros([N])
    theta = np.zeros([N])

    allx[0] = 0.01
    ally[0] = -10
    vx[0] = 0
    vy[0] = 0
    px[0] = 0
    py[0] = 0
    KE[0] = 0
    Us[0] = 0.5*k*(np.sqrt(ally[0]**2 + allx[0]**2) - L)**2
    theta[0] = np.arctan(ally[0]/allx[0])
    Uep[0] = m*g*(np.sqrt(ally[0]**2 + allx[0]**2))*np.cos(theta[0])
    

    for i in range(1,N):
        # Get forces
        Fx, Fy = getForces(m, k, L, allx[i-1], ally[i-1])

        # Update momentum
        px[i] = px[i-1] + Fx/m
        py[i] = py[i-1] + Fy/m

        # Update velocity
        vx[i] = px[i]/m
        vy[i] = py[i]/m

        # Update position
        allx[i] = allx[i-1] + vx[i]*dt
        ally[i] = ally[i-1] + vy[i]*dt

        # Update angle
        theta[i] = np.arctan(ally[i]/allx[i])

        # Update energies of the spring
        KE[i] = 0.5*m*(vx[i]**2+vy[i]**2)
        Us[i] = 0.5*k*(np.sqrt(ally[i]**2 + allx[i]**2) - L)**2
        Uep[i] = m*g*(np.sqrt(ally[i]**2 + allx[i]**2))*np.cos(theta[i])

    return allx, ally, KE, Uep, Us, px, py
        
def getForces(m, k, L, x, y):
    """
    This definition will get the forces of the mass for any specific time
    """
    lmag = np.sqrt(x**2 + y**2)
    s = lmag-L
    xhat = x/lmag
    yhat = y/lmag
    Fy = -9.8*m
    Fy += -k*s*yhat
    Fx = -k*s*xhat
    return Fx, Fy

allx, ally, KE, Uep, Us, px, py = Sim(10,100,5)
#print(allx, ally)

# Make the animation
fig, ax = plt.subplots()
ax = plt.axis([-10,10,-20,0])
point, = plt.plot(allx[0], ally[0], 'ro')
def animate(i):
    point.set_data(allx[i], ally[i])
    return point,
figure = ani.FuncAnimation(fig, animate, interval=10, frames=N)
plt.grid()
plt.show()

plt.figure()
plt.plot(np.arange(0,10,10/N),np.sqrt(px**2 + py**2),label='momentum')
plt.title('Momentum vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Momentum (kg*m/s)')


plt.figure()
plt.plot(np.arange(0,10,10/N),KE,label='Kinetic Energy')
plt.plot(np.arange(0,10,10/N),Uep,label='Elastic Pendulum Potential Energy')
plt.plot(np.arange(0,10,10/N),Us,label='Elastic Potential Energy')
plt.plot(np.arange(0,10,10/N),KE+Uep+Us,label='Total Energy')
plt.title('Energies vs. Time of Elastic Pendulum System')
plt.ylabel('Energy (J)')
plt.xlabel('Time (seconds)')
plt.legend()
plt.show()


# %%