import numpy as np
import matplotlib.pyplot as plt

def grav(M,m,R,r,G):
    """
    Calculate the gravitational force between two masses
    M    is a scalar of one mass
    m    is a scalar of the other mass
    R    is a 1 x 3 matrix of the position of one mass
    r    is a 1 x 3 matrix of the position of the other mass
    G    is the gravitational constant (6.67430e-11 Nm2kg-2)
    """
    dx = R[0] - r[0]
    dy = R[1] - r[1]
    dz = R[2] - r[2]
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    f = np.zeros(3)
    f[0] = G*M*m*dx/d**3
    f[1] = G*M*m*dy/d**3
    f[2] = G*M*m*dz/d**3
    
    return f

def dist(R,r):
    """
    Calculate the distance between two masses
    R    is a 1 x 3 matrix of the position of one mass
    r    is a 1 x 3 matrix of the position of the other mass
    """
    dx = R[0] - r[0]
    dy = R[1] - r[1]
    dz = R[2] - r[2]
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    return d

def ellip1(a, b):
    """
    Calculate ellipticity of a given ellipse using semi-axes
    a    is a scalar of the semi-major axis 
    b    is a scalar of the semi-minor axis
    """
    return np.sqrt(1 - b**2/a**2)

def ellip2(a, p):
    """
    Calculate ellipticity of a given ellipse using semi-major axis and semi latus rectum
    a   is a scalar of the semi-major axis
    p   is a scalar of the semi-latus rectrum
    """
    return np.sqrt(1-p/a)

def slr_for_this(r,R):
    """
    Calculating the position of the Semi latus rectum for this sepcific simulation
    r  is a N x 3 matrix of all the positions that the simulation runs through
    R  is a 1 x 3 matrix of the position of the center
    """
    pot_slr = np.where((r[0:Nt, 0] > -0.001+R[0]) & ((r[0:Nt, 0] < 0.001+R[0]))  )
    r_pot_slr = r[pot_slr]
    slr = np.where( abs(r_pot_slr[0:Nt, 0]-R[0]) == np.max(abs(r_pot_slr[0:Nt, 0]-R[0])))
    r_slr = r_pot_slr[slr]
    return r_slr[0]

def KDK(Nt, dt, r, R, vel, M, m, G):
    """
    Loop for a KDK leap-frog integration
    Nt   is a scalar of the total number of time steps
    dt   is a scalar of a time step
    r    is a N x 3 matrix of the position of one particle
    R    is a 1 x 3 matrix of the position of the center
    vel  is a 1 x 3 matrix of the velocity 
    M    is a scalar of the center mass
    m    is a scalar of the particle mass
    G    is the gravitational constant (6.67430e-11 Nm2kg-2)
    """
    #save positions
    r_save = np.zeros((Nt+1, 3))
    r_save[0] = r
    #save distances
    d_save = np.zeros(Nt+1)
    d_save[0] = dist(R,r)

    for i in range(Nt):
        vel = vel + grav(M,m,R,r,G)/m*dt/2  # 1/2 kick
        r = r + vel*dt  #drift
        vel = vel + grav(M,m,R,r,G)/m*dt/2  # 1/2 kick
        r_save[i+1] = r
        d = dist(R,r)
        d_save[i+1] = d
    return r_save, d_save

def DKD(Nt, dt, r, R, vel, M, m, G):
    """
    Loop for a KDK leap-frog integration
    Nt   is a scalar of the total number of time steps
    dt   is a scalar of a time step
    r    is a N x 3 matrix of the position of one particle
    R    is a 1 x 3 matrix of the position of the center
    vel  is a 1 x 3 matrix of the velocity 
    M    is a scalar of the center mass
    m    is a scalar of the particle mass
    G    is the gravitational constant (6.67430e-11 Nm2kg-2)
    """

    #save positions
    r_save = np.zeros((Nt+1, 3))
    r_save[0] = r
    #save distances
    d_save = np.zeros(Nt+1)
    d_save[0] = dist(R,r)
    
    for i in range(Nt):
        r = r + vel*dt/2  # 1/2 drift
        vel = vel + grav(M,m,R,r,G)/m*dt  #kick
        r = r + vel*dt/2  # 1/2 drift
        r_save[i+1] = r
        d = dist(R,r)
        d_save[i+1] = d
    return r_save, d_save

def second_RK(Nt, dt, r, R, vel, M, m, G):
    """
    Loop for a second order Runge-Kutta integration
    Nt   is a scalar of the total number of time steps
    r    is a N x 3 matrix of the position of one particle
    R    is a 1 x 3 matrix of the position of the center
    vel  is a 1 x 3 matrix of the velocity 
    M    is a scalar of the center mass
    m    is a scalar of the particle mass
    G    is the gravitational constant (6.67430e-11 Nm2kg-2)
    """
    
    #save positions
    r_save = np.zeros((Nt+1, 3))
    r_save[0] = r
    #save distances
    d_save = np.zeros(Nt+1)
    d_save[0] = dist(R,r)
    
    
    for i in range(Nt):
        k_1 = grav(M,m,R,r,G)/m
        h_1 = vel
        k_2 = grav(M,m,R,r+h_1*dt,G)/m
        h_2 = vel+k_1*dt
        vel = vel + 0.5*(k_1+k_2)*dt
        r = r + 0.5*(h_1+h_2)*dt
        r_save[i+1] = r
        d = dist(R,r)
        d_save[i+1] = d
    return r_save, d_save

def fourth_RK(Nt, dt, r, R, vel, M, m, G):
    """
    Loop for a second order Runge-Kutta integration
    Nt   is a scalar of the total number of time steps
    r    is a N x 3 matrix of the position of one particle
    R    is a 1 x 3 matrix of the position of the center
    vel  is a 1 x 3 matrix of the velocity 
    M    is a scalar of the center mass
    m    is a scalar of the particle mass
    G    is the gravitational constant (6.67430e-11 Nm2kg-2)
    """
    
    #save positions
    r_save = np.zeros((Nt+1, 3))
    r_save[0] = r
    #save distances
    d_save = np.zeros(Nt+1)
    d_save[0] = dist(R,r)
    
    for i in range(Nt):
        k_1_v = grav(M,m,R,r,G)/m
        k_1_r = vel
        k_2_v = grav(M,m,R,r+k_1_r*dt/2,G)/m
        k_2_r = vel+k_1_v*dt/2
        k_3_v = grav(M,m,R,r+k_2_r*dt/2,G)/m
        k_3_r = vel+k_2_v*dt/2
        k_4_v = grav(M,m,R,r+k_3_r*dt,G)/m
        k_4_r = vel+k_3_v*dt
        vel = vel + dt/6*(k_1_v+2*k_2_v+2*k_3_v+k_4_v)
        r = r + dt/6*(k_1_r+2*k_2_r+2*k_3_r+k_4_r)
        r_save[i+1] = r
        d = dist(R,r)
        d_save[i+1] = d
    return r_save, d_save

def main():
    """ Orbit simulation """

    #initial conditions
    R = np.array([0,0,0])
    r = np.array([2,0,0])
    M = 10.0
    m = 1.0
    G = 1.0
    vel = np.array([0,1,0])

    #simulation time
    t = 0
    dt = 0.001
    tEnd = 201.5
    Nt = int(np.ceil(tEnd/dt)) #np.ceil rounds up to the next integer

    #run the simulation either KDK or DKD or second-order RK or fourth-order RK
    r_save, d_save = KDK(Nt, dt, r, R, vel, M, m, G)
    #r_save, d_save = DKD(Nt, dt, r, R, vel, M, m, G)
    #r_save, d_save = second_RK(Nt, dt, r, R, vel, M, m, G)
    #r_save, d_save = fourth_RK(Nt, dt, r, R, vel, M, m, G)

    #calculating some data
    #slr = slr_for_this(r_save,R) #position of semi latus rectrum
    #p = dist(R, slr)
    r_apo = np.max(d_save)
    r_peri = np.min(d_save)
    a = (r_apo+r_peri)/2
    #e = ellip2(a,p)
    #ea = e/a

    #prepare data for plotting
    r_save_x = r_save[0:Nt, 0]
    r_save_y = r_save[0:Nt, 1]
    #msg = str(round(ea,4))
    msg = str(round(a,4))
    msg2 = str(dt)
    

    #plotting
    fig = plt.figure()
    fig.set_size_inches(4, 4)
    plt.xlim(-0.8,2.2)
    plt.ylim(-1.5,1.5)
    plt.plot(r_save_x, r_save_y, 'k-')
    plt.plot(r_save_x[0], r_save_y[0], 'yo', label = 'start position')
    plt.plot(r_save_x[Nt-1], r_save_y[Nt-1], 'ro', label = 'end position')
    plt.plot(0,0, 'ko', label = 'fixed center')
    #plt.plot(slr[0],slr[1],'bo', label = 'slr')
    plt.axvline(x=R[0], color = 'gray', linestyle = '--')
    plt.axhline(y=R[1], color = 'gray', linestyle = '--')
    plt.legend(title = 'stepsize = ' + msg2,loc = 4) # + '\n' + 'semi-major axis a = ' + msg
    plt.title('KDK leap-frog integration of an Orbit', fontsize = 15)
    plt.show()
    
    return 0

if __name__== "__main__":
    main()
