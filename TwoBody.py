import numpy as np
import matplotlib.pyplot as plt

def grav(m1,m2,r1,r2,G):
    """
    computes gravitatinal force between two particle
    m1   is a scalar of the mass of 1st particle
    m2   is a scalar of the mass of 2nd particle
    r1   is a 1 x 3 matrix of positons of 1st particle
    r2   is a 1 x 3 matrix of positions of 2nd particle
    G   is the gravitational constant
    """
    dx = r1[0] - r2[0]
    dy = r1[1] - r2[1]
    dz = r1[2] - r2[2]
    d = np.sqrt(dx**2 + dy**2 + dz**2)
    f = np.zeros(3)
    f[0] = G*m1*m2*dx/d**2
    f[1] = G*m1*m2*dy/d**2
    f[2] = G*m1*m2*dz/d**2
    return f

def KDK_two_body(Nt, dt, r1, r2, vel1, vel2, m1, m2, G):
    """
    loop for a KDK leap frog integration of a two body problem
    Nt   is a scalar of the number of the total time steps
    dt   is a scalar of the size of the time step
    r1   is a 1 x 3 matrix of positons of 1st particle
    r2   is a 1 x 3 matrix of positions of 2nd particle
    vel1 is a 1 x 3 matrix of velocity of the 1st particle
    vel2 is a 1 x 3 matrix of velocity of the 2nd particle
    m1   is a scalar of the mass of 1st particle
    m2   is a scalar of the mass of 2nd particle
    G   is the gravitational constant
    """
    #save positions
    r1_save = np.zeros((Nt+1, 3))
    r2_save = np.zeros((Nt+1, 3))
    r1_save[0] = r1
    r2_save[0] = r2    
    
    for i in range(Nt):
        vel1 = vel1 + grav(m2,m1,r2,r1,G)/m1*dt/2
        vel2 = vel2 + grav(m1,m2,r1,r2,G)/m2*dt/2
        
        r1 = r1 + vel1*dt
        r2 = r2 + vel2*dt
        
        vel1 = vel1 + grav(m2,m1,r2,r1,G)/m1*dt/2
        vel2 = vel2 + grav(m1,m2,r1,r2,G)/m2*dt/2
        
        r1_save[i+1] = r1
        r2_save[i+1] = r2
        
    return r1_save, r2_save

def main():
    #simulation data
    m1 = 1000.0 
    m2 = 1.0
    r1 = np.array([0.0,0,0])
    r2 = np.array([0.0,1.0,0])
    vel1 = np.array([10.0,0.0,0.0])
    vel2 = np.array([0,0.0,5.0])
    G = 1.0

    #simulation time
    t = 0
    dt = 0.001
    tEnd = 0.5
    Nt = int(np.ceil(tEnd/dt)) #np.ceil rounds up to the next integer

    r1_save, r2_save = KDK_two_body(Nt, dt, r1, r2, vel1, vel2, m1, m2, G)
    #r1_save = fourth_RK_two_body(Nt, dt, r1, r2, vel1, vel2, m1, m2, G)

    #prepare data for plotting
    r1_save_x = r1_save[0:Nt, 0]
    r1_save_y = r1_save[0:Nt, 1]
    r1_save_z = r1_save[0:Nt, 2]
    r2_save_x = r2_save[0:Nt, 0]
    r2_save_y = r2_save[0:Nt, 1]
    r2_save_z = r2_save[0:Nt, 2]
    

    #plotting
    fig = plt.figure()

    #line of sight is z axis
    #plt.plot(r1_save_x, r1_save_y, c = 'orange', ls = '-')
    #plt.plot(r2_save_x, r2_save_y, c = 'gold', ls = '-')
    #plt.plot(r1_save_x[0], r1_save_y[0], 'rv', label = 'start position m1')
    #plt.plot(r1_save_x[Nt-1], r1_save_y[Nt-1], 'ro', label = 'end position m1')
    #plt.plot(r2_save_x[0], r2_save_y[0], 'yv', label = 'start position m2') #
    #plt.plot(r2_save_x[Nt-1], r2_save_y[Nt-1], 'yo', label = 'end position m2')

    #line of sight is x-axis
    plt.plot(r1_save_z, r1_save_y, c = 'orange', ls = '-')
    plt.plot(r2_save_z, r2_save_y, c = 'gold', ls = '-')
    plt.plot(r1_save_z[0], r1_save_y[0], 'rv', label = 'start position m1')
    plt.plot(r1_save_z[Nt-1], r1_save_y[Nt-1], 'ro', label = 'end position m1')
    plt.plot(r2_save_z[0], r2_save_y[0], 'yv', label = 'start position m2') #
    plt.plot(r2_save_z[Nt-1], r2_save_y[Nt-1], 'yo', label = 'end position m2')

    
    plt.legend(loc = 'lower center')
    plt.title('KDK leap-frog integration of two bodies', fontsize = 20 )
    plt.show()
    
    return 0

if __name__== "__main__":
    main()