"""
Tutorial for the calculation of exciton radiative lifetimes.

Plotting the radiative lifetime of the lowest-energy exciton of GaN.
Plotting the thermal average exciton radiative lifetime of GaN.

"""
from yambopy import *
import numpy as np
import matplotlib.pyplot as plt


# modify path accordingly 
lattice_path='./'
bse_path ='./'
dipoles_path ='./'

if __name__ == "__main__":

    #                    #
    # Start Yambopy part #
    #                    #

    print('Loading databases')
    # Create "lattice" object by reading the ns.db1 database
    ylat = YamboLatticeDB.from_db_file(filename=lattice_path+'/ns.db1')
    # Read exciton data
    yexc = YamboExcitonDB.from_db_file(ylat,filename=bse_path+'/ndb.BS_diago_Q1')
    # Read dipoles
    ydip = YamboDipolesDB.from_db_file(ylat,filename=dipoles_path+'/ndb.dipoles',project=False)
    print('     --- done.')


    # Provide the exciton effective masses and dielectric constants.
    #
    # GaN is a uniaxial 3D bulk material, with in-plane (xy) and out-of-plane (y) anisotropy
    #
    # Exciton effective masses in units of the electron rest mass
    M_xy = 2.5796  # m_e units
    M_z = 2.8778   # m_e units

    # Dielectric constants
    eps_xy = 6.852
    eps_z = 7.005

    # Define the temperature range

    Trange = np.arange(10,273,1)


    # Get the radiative lifetime of exciton state=0 in seconds
    print('Computing exciton state=0 radiative lifetime')
    state = 0
    tau_0 = get_radiative_lifetime_3D_aniso(Trange,state,ylat,ydip,yexc,[M_xy,M_z],[eps_xy,eps_z]) # seconds


    print('     Plotting')
    fig,ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$\tau_{S=0}$ (s)')
    ax.plot(Trange,tau_0)
    ax.set_title('Radiative lifetime of exciton state $S=0$ in GaN')

    fig.savefig('GaN_exciton_0_radiative_lifetime.png',dpi=200,bbox_inches='tight')
    plt.show()
    print('     --- done')


    # Get the thermal average radiative lifetime of GaN in seconds
    #
    # Define the exciton states included in the average
    print('Computing average exciton radiative lifetime')
    states = range(10)
    tau_avg = average_lifetime(Trange,states,ylat,ydip,yexc,'3D_aniso',[M_xy,M_z],[eps_xy,eps_z])  # seconds

    print('     Plotting')
    fig,ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$\langle\tau\rangle$ (s)')
    ax.plot(Trange,tau_avg)
    ax.set_title('Average exciton radiative lifetime of GaN')

    fig.savefig('GaN_exciton_avg_radiative_lifetime.png',dpi=200,bbox_inches='tight')
    plt.show()
    print('     --- done')
    print('All done')



    #                  #
    # End Yambopy part #
    #                  #
