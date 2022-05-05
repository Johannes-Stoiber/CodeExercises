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
    r_save = np.zeros((Nt, 3))
    #save distances
    d_save = np.zeros(Nt)

    for i in range(Nt):
        r_save[i] = r
        d = dist(R,r)
        d_save[i] = d
        vel = vel + grav(M,m,R,r,G)/m*dt/2  # 1/2 kick
        r = r + vel*dt  #drift
        vel = vel + grav(M,m,R,r,G)/M*dt/2  # 1/2 kick
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
    r_save = np.zeros((Nt, 3))
    #save distances
    d_save = np.zeros(Nt)
    
    for i in range(Nt):
        r_save[i] = r
        d = dist(R,r)
        d_save[i] = d
        r = r + vel*dt/2  # 1/2 drift
        vel = vel + grav(M,m,R,r,G)/m*dt  #kick
        r = r + vel*dt/2  # 1/2 drift
    return r_save, d_save

def main():
    """ Orbit simulation """

    #initial conditions
    R = np.array([0,0,0])
    r = np.array([1,0,0])
    M = 10.0
    m = 1.0
    G = 1.0
    vel = np.array([0,2,0])

    #simulation time
    t = 0
    dt = 0.01
    tEnd = 10.0
    Nt = int(np.ceil(tEnd/dt))# np.ceil rounds up to the next integer

    #run the simulation either KDK or DKD
    #r_save, d_save = KDK(Nt, dt, r, R, vel, M, m, G)
    r_save, d_save = DKD(Nt, dt, r, R, vel, M, m, G)

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

    #plotting
    fig = plt.figure()
    plt.plot(r_save_x, r_save_y, 'k-')
    plt.plot(r_save_x[0], r_save_y[0], 'yo', label = 'start position')
    plt.plot(r_save_x[Nt-1], r_save_y[Nt-1], 'ro', label = 'end position')
    plt.plot(0,0, 'ko', label = 'fixed center')
    #plt.plot(slr[0],slr[1],'bo', label = 'slr')
    plt.axvline(x=R[0], color = 'gray', linestyle = '--')
    plt.axhline(y=R[1], color = 'gray', linestyle = '--')
    plt.legend(loc = 1)
    plt.title('KDK leap-frog-integration of an Orbit with a = ' + msg)
    plt.show()
    
    return 0

if __name__== "__main__":
    main()
