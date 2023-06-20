
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
import os
import time
import subprocess
import matplotlib

import numpy                as np
import seaborn              as sns
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec

# ---------------------------
#     Code configuration
# ---------------------------
# Determine if the Serial version should be tested
test_SERIAL = True

# Determine if OpenMP version should be tested
test_OPENMP = True

# Determine if CUDA version should be tested
test_CUDA = True

# Determine if CUDA with GRAPHS version should be tested
test_CUDA_GRAPHS = False

# ---------------------------
#   Simulation parameters
# ---------------------------
# Total number of times the code is executed over the same simulation
nb_times = 5

# Simulation duration
max_t = 0.2

# Time step
delta_t = 0.1

# Number of threads used by OpenMP
nb_threads = 48

# Figure size
fig_size = (15.2, 8)

# List of problems to solve
prob_list = [f"galaxy_{i}" for i in range(1, 21)]

# Size of the problems, i.e. number of particles per problem
prob_size = [1000 * i for i in range(1, 21)]

# Contains all the colors associated to each executables
colors = {
    "NBODY_SERIAL_LEFT"      :(0, 0, 0, 0.6),
    "NBODY_SERIAL_RIGHT"     :(0, 0, 0, 0.6),
    "NBODY_OPENMP_LEFT"      :(245/255, 0, 0, 0.6),
    "NBODY_OPENMP_RIGHT"     :(245/255, 0, 0, 0.6),
    "NBODY_CUDA_LEFT"        :(0, 153/255, 76/255, 0.6),
    "NBODY_CUDA_RIGHT"       :(0, 153/255, 76/255, 0.6),
    "NBODY_GRAPH_CUDA_LEFT"  :(0, 0, 204/255, 0.6),
    "NBODY_GRAPH_CUDA_RIGHT" :(0, 0, 204/255, 0.6),
}

# ---------------------
# Preparing executables
# ---------------------
exec_cors, exec_list, exec_name = list(), list(), list()

if test_SERIAL == True:
    exec_cors += [1, 1]
    exec_list += ["NBODY_SERIAL_RIGHT", "NBODY_SERIAL_LEFT"]
    exec_name += [r"$Serial_{\ \bf{R}}$", r"$Serial_{\ \bf{L}}$"]

if test_OPENMP == True:
    exec_name_left  = str(nb_threads) + r", \bf{L}"
    exec_name_right = str(nb_threads) + r", \bf{R}"
    exec_cors += [4, 4]
    exec_list += [ "NBODY_OPENMP_RIGHT", "NBODY_OPENMP_LEFT"]
    exec_name += [r"$OpenMP_{\ " + exec_name_right + r"}$", r"$OpenMP_{\ " + exec_name_left + r"}$" ]

if test_CUDA == True:
    exec_cors += [5120, 5120]
    exec_list += ["NBODY_CUDA_RIGHT", "NBODY_CUDA_LEFT"]
    exec_name += [r"$Cuda_{\ \bf{R}}$", r"$Cuda_{\ \bf{L}}$"]

if test_CUDA_GRAPHS == True:
    exec_cors += [5120, 5120]
    exec_list += ["NBODY_GRAPH_CUDA_RIGHT", "NBODY_GRAPH_CUDA_LEFT"]
    exec_name += [r"$Cuda_{\ GRAPH, \bf{R}}$", r"$Cuda_{\ GRAPH, \bf{L}}$"]

# Security for dummies
assert len(exec_list) == len(exec_cors), "/!\ Exec_list and Exec_cors must be of same length "
assert len(prob_list) == len(prob_size), "/!\ Prob_list and Prob_size must be of same length "
assert len(exec_list) > 0,               "/!\ Should have at least one executable ! "

# -------------------
#        Main
# -------------------
if __name__ == '__main__':

    # Store the different timings and bandwidth
    time_seq = np.zeros([len(exec_list), len(prob_list)], dtype = float)
    time_par = np.zeros([len(exec_list), len(prob_list)], dtype = float)
    time_tot = np.zeros([len(exec_list), len(prob_list)], dtype = float)
    sim_bandwidth = np.ones([len(exec_list), len(prob_list)])

    # Store the error made on the measurement
    time_seq_error      = np.zeros([len(exec_list), len(prob_list)], dtype = float)
    time_par_error      = np.zeros([len(exec_list), len(prob_list)], dtype = float)
    time_tot_error      = np.zeros([len(exec_list), len(prob_list)], dtype = float)
    sim_bandwidth_error = np.ones([len(exec_list), len(prob_list)])

    # Post-fix information to save data
    info = f"{max_t}_{delta_t}_{nb_times}"

    # Stores the bandwidth of each simulation
    sim_bandwidth = np.ones([len(exec_list), len(prob_list)])

    for i, exe in enumerate(exec_list):

        # Displaying information over terminal (1)
        print("Executable :", exe)

        # Adaptating number of threads to use
        if "SERIAL" in exe:
            os.environ["OMP_NUM_THREADS"] = "1"
        else:
            os.environ["OMP_NUM_THREADS"] = "48"

        for j, prob in enumerate(prob_list):

            # Displaying information over terminal (1)
            print(" - Problem :", prob)

            # Used to compute the errors made
            tseq_error, tpar_error, ttot_error, band_error = list(), list(), list(), list()

            for t in range(nb_times):

                # Displaying information over terminal (3)
                print("     - Iter :", t)

                # Check if corresponding executable folder exists
                if not os.path.isdir(f"datasets/{exe}"):
                    os.mkdir(f"datasets/{exe}")

                # Check if corresponding simulation folder exists
                if not os.path.isdir(f"datasets/{exe}/{prob}"):
                    os.mkdir(f"datasets/{exe}/{prob}")

                # Check if data already exist
                if os.path.isfile(f"datasets/{exe}/{prob}/time_seq_{info}.npy"):
                    time_seq[i][j]            = np.load(f"datasets/{exe}/{prob}/time_seq_{info}.npy")
                    time_par[i][j]            = np.load(f"datasets/{exe}/{prob}/time_par_{info}.npy")
                    time_tot[i][j]            = np.load(f"datasets/{exe}/{prob}/time_tot_{info}.npy")
                    sim_bandwidth[i][j]       = np.load(f"datasets/{exe}/{prob}/bandwidth_{info}.npy")
                    time_seq_error[i][j]      = np.load(f"datasets/{exe}/{prob}/time_seq_error_{info}.npy")
                    time_par_error[i][j]      = np.load(f"datasets/{exe}/{prob}/time_par_error_{info}.npy")
                    time_tot_error[i][j]      = np.load(f"datasets/{exe}/{prob}/time_tot_error_{info}.npy")
                    sim_bandwidth_error[i][j] = np.load(f"datasets/{exe}/{prob}/bandwidth_error_{info}.npy")
                    break

                # Running the simulation
                sim = subprocess.run([f"./build/{exe} -file data/{prob}.txt -maxt {max_t} -deltat {delta_t}"],
                                      shell = True, capture_output = True, text = True)

                # Displaying information over terminal
                print(sim.stdout)

                # Extracting sequential time [s]
                time_seq[i][j] += float(sim.stdout.split()[4].replace("[s]", ""))

                # Extracting parallelize time [s]
                time_par[i][j] += float(sim.stdout.split()[9].replace("[s]", ""))

                # Extracting total time [s]
                time_tot[i][j] += float(sim.stdout.split()[14].replace("[s]", ""))

                # Extrating the bandwidth [GB/s]
                sim_bandwidth[i][j] += float(sim.stdout.split()[17].replace("[GB/s]", ""))

                # Adding values to compute errors later on
                tseq_error.append(float(sim.stdout.split()[4].replace("[s]", "")))
                tpar_error.append(float(sim.stdout.split()[9].replace("[s]", "")))
                ttot_error.append(float(sim.stdout.split()[14].replace("[s]", "")))
                band_error.append(float(sim.stdout.split()[17].replace("[GB/s]", "")))

            # Averaging and saving data if not already done
            if not os.path.isfile(f"datasets/{exe}/{prob}/time_seq_{info}.npy"):

                # Averaging and conversion to seconds
                time_seq[i][j] = time_seq[i][j]/(nb_times)
                time_par[i][j] = time_par[i][j]/(nb_times)
                time_tot[i][j] = time_tot[i][j]/(nb_times)

                # Averaging the bandwidth
                sim_bandwidth[i][j] = sim_bandwidth[i][j]/nb_times

                # Computing errors
                time_seq_error[i][j]      = np.std(tseq_error)
                time_par_error[i][j]      = np.std(tpar_error)
                time_tot_error[i][j]      = np.std(ttot_error)
                sim_bandwidth_error[i][j] = np.std(band_error)
                print("ICI :", sim_bandwidth_error[i][j])

                # Saving
                np.save(f"datasets/{exe}/{prob}/time_seq_{info}",        time_seq[i][j])
                np.save(f"datasets/{exe}/{prob}/time_par_{info}",        time_par[i][j])
                np.save(f"datasets/{exe}/{prob}/time_tot_{info}",        time_tot[i][j])
                np.save(f"datasets/{exe}/{prob}/bandwidth_{info}",       sim_bandwidth[i][j])
                np.save(f"datasets/{exe}/{prob}/time_seq_error_{info}",  time_seq_error[i][j])
                np.save(f"datasets/{exe}/{prob}/time_par_error_{info}",  time_par_error[i][j])
                np.save(f"datasets/{exe}/{prob}/time_tot_error_{info}",  time_tot_error[i][j])
                np.save(f"datasets/{exe}/{prob}/bandwidth_error_{info}", sim_bandwidth_error[i][j])

    # ------------------
    # Results & Plotting
    # ------------------
    # Change plot theme
    sns.set_theme(style = "darkgrid")
    sns.color_palette("deep")

    # Total number of plots in the benchmark folder
    nb_file = int(len(os.listdir("benchmarks"))/3)

    # Total number of iterations made by simulation
    n_iter = max_t/delta_t

    # Stores the speed-up of each simulation
    speed_up = np.ones([len(exec_list), len(prob_list)])

    # Displaying information over terminal
    for i, exe in enumerate(exec_list):
        for j, prob in enumerate(prob_list):
            print(f"\n{exe} results in [s] on {prob} :")
            print("Sequential   = ", time_seq[i][j], " [s]")
            print("Parallelized = ", time_par[i][j], " [s]")
            print("Total        = ", time_tot[i][j], " [s]")
            print("Bandwidth    = ", sim_bandwidth[i][j], " [GB/s]")

    # Increases overall fontsize
    plt.rcParams.update({'font.size': 15})

    # ---------- SPEED UP -----------
    #
    # Initalization of the plot
    fig = plt.figure(figsize = fig_size)
    plt.yscale('log')

    # Plotting speed up
    for i, exe in enumerate(exec_name):

        # Stores the upper and lower bound error of the speed up
        speed_up_error = np.zeros([2, len(prob_list)], dtype = float)

        # Computing speed up
        speed_up[i][:] = time_par[0][:]/time_par[i][:]

        # Lower error (i = 0, see documentation)
        speed_up_error[0][:] = np.abs(time_par[0][:]/(time_par[i][:] + time_par_error[i][:]) - speed_up[i][:])

        # Upper error (i = 1, see documentation)
        speed_up_error[1][:] = np.abs(time_par[0][:]/(time_par[i][:] - time_par_error[i][:]) - speed_up[i][:])

        # Plotting results
        plt.errorbar(prob_size, speed_up[i][:], yerr = speed_up_error, label = exe, lw = 1.5, marker = 'o', markersize = 4,
                     fmt = '-' if i%2 == 0 else '--', elinewidth = 2, color = colors[exec_list[i]])

    # Adding ylabel
    plt.ylabel("Speed-UP [-]", fontsize = 16, labelpad = 15)

    # Adding a x-label
    plt.xlabel("Number of particles [-]", fontsize = 16, labelpad = 15)

    # Adding legend
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol = len(exec_name), framealpha = 1, loc = "lower left")

    # Adding nice grid
    plt.grid(which = 'both', alpha = 0.6)

    # Saving the figure
    plt.savefig(f"benchmarks/benchmarks_{str(nb_file)}_speedup.png", bbox_inches = "tight")

    # Closing current figure
    plt.close()

    # ---------- EFFICIENCY -----------
    #
    # Initalization of the plot
    fig = plt.figure(figsize = fig_size)
    plt.yscale('log')

    # Plotting speed up
    for i, exe, exe_core in zip(range(len(exec_list)), exec_name, exec_cors):
        plt.semilogy(prob_size, speed_up[i][:]/exe_core, label = exe, lw = 2, marker = 'o', markersize = 4,
                     linestyle = 'solid' if i%2 == 0 else 'dashed')
    # Adding ylabel
    plt.ylabel("Efficiency [-]", fontsize = 16, labelpad = 15)

    # Adding a x-label
    plt.xlabel("Number of particles [-]", fontsize = 16, labelpad = 15)

    # Adding legend
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol = len(exec_name), framealpha = 1, loc = "lower left")

    # Adding nice grid
    plt.grid(which = 'both', alpha = 0.6)

    # Saving the figure
    plt.savefig(f"benchmarks/benchmarks_{str(nb_file)}_efficiency.png", bbox_inches = "tight")

    # Closing current figure
    plt.close()

    # ---------- BANDWIDTH ----------
    #
    # Initalization of the plot
    fig = plt.figure(figsize = fig_size)
    plt.yscale('log')

    # Plotting bandwidth
    for i, exe in enumerate(exec_name):

        # Error
        plt.errorbar(prob_size, sim_bandwidth[i][:], yerr = sim_bandwidth_error[i][:], label = exe, lw = 2, marker = 'o', markersize = 4,
                     fmt = '-' if i%2 == 0 else '--', elinewidth = 1, color = colors[exec_list[i]])

    # Adding ylabel
    plt.ylabel("Bandwidth [GB/s]", fontsize = 16, labelpad = 15)

    # Adding legend
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol = len(exec_name), framealpha = 1, loc = "lower left")

    # Adding nice grid
    plt.grid(which = 'both', alpha = 0.6)

    # Adding a x-label
    plt.xlabel("Number of particles [-]", fontsize = 16, labelpad = 15)

    # Saving the figure
    plt.savefig(f"benchmarks/benchmarks_{str(nb_file)}_bandwidth.png", bbox_inches = "tight")

    # Closing current figure
    plt.close()