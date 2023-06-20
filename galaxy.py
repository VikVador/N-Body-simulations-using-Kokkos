
# --------------------------------------------------------------------------------
#
#                        New Methods in Computational Physics
#
#                                N - Body Simulation
#
# -----------------------                             ----------------------------
# Arnaud Remi - S183416 |                             | Victor Mangeleer - S181670
# --------------------------------------------------------------------------------
#
# ---------------------------
#         Librairies
# ---------------------------
import random
import subprocess
import numpy as np

# ---------------------------
#       Data parameters
# ---------------------------
# Define the name of the simulation file
simulation_name = "galaxy_triangle"

# Define if the simulation should be run or not (using GPU and results saved directly)
simulation_run = True

# Define simulation duration
max_t = 20

# Define simulation time step
delta_t = 0.1

# Define if the simulation results should be transformed in a gif
simulation_gif = True

# ---------------------------
#         Functions
# ---------------------------
def create_Galaxy(file, nb_particles, body_type, black_hole_center, black_hole_mass, mass_min, mass_max, radius_min, radius_max, speed_coef):

        # --- Black Hole Information ---
        file.write(str(black_hole_center[0]) + ' ' + str(black_hole_center[1]) + ' ' + str(0.) + ' ' + str(0.) + ' ' + str(black_hole_mass) + '\n')

        # --- Galaxy Type 1 - Elliptical ---
        if 'Elliptic' in body_type:
            for i in range(nb_particles):

                # Computing star information
                r     = random.uniform(radius_min, radius_max)
                theta = random.uniform(0, 2 * np.pi)
                x     = r * np.cos(theta) - black_hole_center[0]
                y     = r * np.sin(theta) - black_hole_center[1]
                m     = random.uniform(mass_min, mass_max)
                vx    = - speed_coef * (G * black_hole_mass / r) ** 0.5 * np.sin(theta)
                vy    = + speed_coef * (G * black_hole_mass / r) ** 0.5 * np.cos(theta)

                # Adding star information
                file.write(str(x) + ' ' + str(y) + ' ' + str(vx) + ' ' + str(vy) + ' ' + str(m) + '\n')

        # --- Galaxy Type 2 - Circular ---
        elif 'Circular' in body_type:

            # TO BE DONE ?
            pass

        else:
            pass



def create_AsteroidBelt():
    pass

def create_SpatialShip():
    pass

# ---------------------------
#            Main
# ---------------------------
#
# -------- PHASE 1 : Simulation file --------
#
# Opening simulation file
with open(f"data/{simulation_name}.txt", 'w') as file:

    # --- Parameters (fixed) ---
    #
    # Cavendish constant
    G = 6.74e-11

    # --- Parameters (each celestial body) ---
    #
    # Number of particles
    k = [4000, 4000, 100]

    # Black-holes mass
    bh_mass = [1e31, 1e33, 1e32]

    # Black-holes positions
    bh_position = [[    0,   0],
                   [ 10e8,   0],
                   [  5e8, 5e8]]

    # Outer and inner radius of the galaxy belt
    radius_max = [4e8, 4e8, 6e8]
    radius_min = [3e6, 3e6, 5.99e8]

    # Mass interval of stars inside the belt
    mass_max = [1e2, 10e2, 5e30]
    mass_min = [3e1, 4e1,  2e30]

    # Speed amplification coefficient
    speed_coeff = [3, 3, 0.00001]

    # Type of celestial body
    body_type = ["Galaxy - Elliptic", "Galaxy - Elliptic", "Galaxy - Elliptic"]

    # --- Number of particles ---
    file.write(str(sum(k) + len(k)) + '\n')

    # --- Celestial Bodies ---
    for k_i, b_type, bh_m, bh_pos, r_min, r_max, m_min, m_max, speed \
    in zip(k, body_type, bh_mass, bh_position, radius_min, radius_max, mass_min, mass_max, speed_coeff):

        if "Galaxy" in b_type:
            create_Galaxy(file, k_i, b_type, bh_pos, bh_m, m_min, m_max, r_min, r_max, speed)
        else:
            # More functions can be added here !
            pass

print("Phase 1 - Done")
#
# -------- PHASE 2 : Simulation --------
#
if simulation_run:
    sim = subprocess.run([f"./build/NBODY_SIMULATION -file data/{simulation_name}.txt -maxt {max_t} -deltat {delta_t}"], shell = True, capture_output = True, text = True)

print("Phase 2 - Done")
#
# -------- PHASE 2 : Simulation visialization --------
#
if simulation_gif:
    sim = subprocess.run([f"python3 vizualize.py"], shell = True, capture_output = True, text = True)

print("Phase 3 - Done")