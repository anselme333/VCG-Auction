# Interpreter: python 2.7.14
########################################################################################################################
# Needed packages
from __future__ import division

from plotly.offline.offline import matplotlib
from random_words import RandomWords
# RandomWords is a useful package for generate random content name for LFU caching
# from pyunlocbox import functions, solvers #https://pypi.python.org/pypi/pyunlocbox
import random
import math
from scipy.interpolate import spline
import seaborn
import simpy
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
import matplotlib.pyplot as plt; plt.rcdefaults()
import itertools
plt.style.use('ggplot')
from scipy import special
import cvxpy as cvx
from gurobipy import *
from matplotlib.ticker import FuncFormatter
import dmcp
from dmcp.fix import fix
from dmcp.find_set import find_minimal_sets
from dmcp.bcd import is_dmcp
import lfucache.lfu_cache as lfu_cache  # LFU replacement policy from https://github.com/laurentluce/lfu-cache
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
import BNG
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
seaborn.set_style(style='white')
from scipy import special
########################################################################################################################
########################################################################################################################

startTime = time.time()
# Remove outliers
"""
 OKM for collaboration space (OKM-CS)
 Forming collaboration space:
 Source of the code: http://all-geo.org/volcan01010/2012/11/change-coordinates-with-pyproj/
 https://finds.org.uk/getinvolved/guides/ngr
 Source of dataset:  https://www.ofcom.org.uk/phones-telecoms-and-internet/coverage/mobile-operational-enquiries
"""


def reject_outliers_2(data, m =2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/(mdev if mdev else 1.)
    return data[s < m]


# Region codes for 100 km grid squares.
col_names = ['Operator', 'Opref', 'Sitengr', 'Antennaht', 'Transtype', 'Freqband', 'Anttype','Powerdbw', 'Maxpwrdbw', 'Maxpwrdbm', 'Sitelat', 'Sitelng']
import_data = pd.read_csv('C:/Users/anselme/Google Drive/research/Simulation_Research/Journal3/simulation_4C_02/SITEFINDER_MAY_2012_VODAPHONE.csv', names=col_names,)
bs_data = import_data.values
location_bs = bs_data[:, 2]
location_bs = location_bs.tolist()
location_bs_array = []
number_bs = len(location_bs)
xy = BNG.to_osgb36(location_bs)
x, y = zip(*xy)
x_coordinate = np.array(x)
y_coordinate = np.array(y)
import_data["x_coordinate"] = x_coordinate
import_data["y_coordinate"] = y_coordinate
data_set_with_xy = import_data.to_csv('SITEFINDER_MAY_2012_VODAPHONE_COORDINATES.csv')


# Import data
df = pd.read_csv('SITEFINDER_MAY_2012_VODAPHONE_COORDINATES.csv', low_memory=False, delimiter=',')
x_coordinate = df["x_coordinate"]
y_coordinate = df["y_coordinate"]

x_coordinate = np.array(x_coordinate)
# x_coordinate = reject_outliers_2(x_coordinate)

y_coordinate = np.array(y_coordinate)
# y_coordinate = reject_outliers_2(y_coordinate)


coordinate = list(zip(x_coordinate, y_coordinate))

##############

distortions = []
for i in range(1, 500):
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(coordinate)
    distortions.append(km.inertia_)
plt.plot(range(1, 500), distortions, marker='o')
plt.xlabel('Number of clusters', fontsize=18)
plt.ylabel('Distortion', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(0, 10000)
plt.tight_layout()
plt.grid(color='gray', linestyle='dashed')
plt.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal3/simulation_4C_02/plots/"
            "elbow_clusters.pdf",  bbox_inches='tight')
plt.show()

##############

print(" OKM for collaboration space ")
K = 1000
kmeans = KMeans(n_clusters=K)
kmeans.fit(coordinate)
print("coordinate length", len(coordinate))
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
number_bs_cluster = dict(Counter(labels))
average_number_bs = np.mean(number_bs_cluster.values())
max_number_bs_cluster = max(number_bs_cluster.values())  # maximum number of BS per cluster
min_number_bs_cluster = min(number_bs_cluster.values())  # minimum number of BS per cluster

print "everage_number_bs", average_number_bs
print "max_number_bs_cluster", max_number_bs_cluster
print "min_number_bs_cluster", min_number_bs_cluster
colors = ["g.", "m.", "c.", "y.", "b.", "m.", "k.", "y.", "b.", "g.", "c.", "m.", "b.", "y.", "k.", "g."]
fig1 = plt.figure()
for i in range(len(coordinate)):
    k = labels[i]/100
    k = int(k)
    print("coordinate:", coordinate[i], "label:", labels[i])
    plt.plot(coordinate[i][0], coordinate[i][1], colors[k], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1],  color='red', marker="x", s=150, linewidths=5, zorder=10, label='Centroids')
plt.grid(color='gray', linestyle='dashed')
plt.xlabel('X coordinates', fontsize=18)
plt.ylabel('Y coordinates', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
plt.show()
endTime0 = time.time()
simulation_time = endTime0 - startTime
print "Simulation time for clustering:", simulation_time, "seconds"
fig1.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal3/simulation_4C_02/"
             "plots/clusters_bs.pdf", bbox_inches='tight')

##########################################################################################################

#  Initial  settings:
env = simpy.Environment()
RANDOM_SEED = 1000
random.seed(RANDOM_SEED)
number_mec_server = int(average_number_bs)# Number of MEC servers in collaborative MEC
path_loss_factor = 4        # Path loss exponents
number_channel = 5          # Number of channel
channel_bandwidth = 20      # Channel bandwidth in terms of Mhz
Bandwidth_BS_m_BS_n = random.randint(20, 25)     # Bandwidth  between BS m and BS n in terms of Mbps
Bandwidth_BS_m_cloud = random.randint(50, 120)   # Bandwidth  between BS m and BS n in terms of Mbps
transmission_power = 27.0   # Transmission power in term of dbm of each users
number_iteration = 100      # Initialization of the number of iteration
spectrum_allocation = {}    # Spectrum allocation variables
ground_distance = 40        # Distance between end-user and BS in terms of metres
distance_BS_user = 20       # Distance between end-user and BS in terms of metres
Fading = 3                  # Rayleigh  Fading
end_user = 100              # initial end-users
active_user_list = []
frequency = {}
local_computation_cost = []    # Local computation cost
transm_delay =[]
########################################################################################################################

# At end-user device
number_application = random.randint(2, 5)         # Number of applications at each  end-user device
cpu_energy = 3                                    # Energy at end-user device in terms W/GhZ
computation_deadline = random.uniform(0.02, 12)   # computation deadline in terms of seconds
user_cache_allocation_variable = []               # cache decision variable
end_user_device_stutus = []                       # End_user device status
user_offloading_variable = []                     # computation decision variable
cpu_arc_parameter = 2                             # constant parameter that is related to CPU hardware architecture
cache_capacity_allocation_user = []
computation_capacity_allocation_user = []
global execution_latency_user
global transm_delay_user_bs
percentage_radio_spectrum_vector = []
input_data_vector = []
user_execution_latency_array = []
output_data = []
instantaneous_data_vector = []
########################################################################################################################
# At mec server
p_mec = random.randint(2000000000, 2500000000)   # Computation capacity at each MEC in terms of Cycles Per Second                                              # (from 2 GHz to 2.5 GHz)
c_mec = random.uniform(100000, 500000)           # Storage capacity at each end-user device in terms of gigabyte
MEC_m_cache_allocation_variable = []             # cache decision variable
MEC_n_cache_allocation_variable = []             # cache decision variable
MEC_m_computation_allocation_variable = []       # cache decision variable
MEC_n_computation_allocation_variable = []       # cache decision variable
MEC_m_MEC_n_offloading_variable = []             # computation offloading (MEC_m_MEC_n) decision variable
MEC_m_cloud_offloading_variable = []             # computation offloading (MEC_m_cloud) decision variable
cache_capacity_allocation_MEC = []               # Computation allocation  at MEC
computation_capacity_allocation_MEC = []         # Cache  allocation  at MEC
revenue_radio_resource_vector = []               # Radio resource vector
total_executing_mc_m_array = []
total_executing_mc_n_array = []
content_name_mc = []
MEC_server_maximum_capacity_caching = []         # Resources for MEC servers
# Caching resources in terms of gigabyte
# Caching resources in terms of gigabyte
caching_resource = [random.randint(100000, 500000) for _ in range(number_mec_server)]
# Computation resources in terms of Cycles Per Second
computation_resource = [random.randint(2000000000, 2500000000) for _ in range(number_mec_server)]
collaboration_space_capacity = [random.randint(2000000000, 2500000000) for _ in range(number_mec_server)]
MEC_server_maximum_capacity_computation = []
transm_delay_between_bs = []
#######################################################################################################################
# At data center
p_d = random.randint(2500000000, 2900000000)    # Computation  capacity at each end-user in terms of Cycles Per Second
                                                # (from 2.5 GHz to 3.9 GHz)
c_d = random.uniform(100000, 500000)            # Storage capacity at each end-user device in terms of gigabyte
cache_capacity_allocation_dc = []               # Cache  allocation  at DC
computation_capacity_allocation_dc = []         # Computation allocation at DC
DC_caching_decision_variable = []               # Cache decision variable
DC_computation_decision_variable = []           # Cache decision variable
execution_latency_dc_vector = []                # Execution latency_dc
cache_dc = lfu_cache.Cache()
number_dc_server = 20                           # Number of servers in DC


class DataCenter(object):

    def __init__(self, number_dc_server, end_user, input_data, computation_deadline, computation_requirement_user
                 , p_d, c_d):
        self.number_dc_server = number_dc_server
        self.p_d = p_d
        self.c_d = c_d
        self.input_data = input_data
        self.computation_deadline = computation_deadline
        self.computation_requirement_user = computation_requirement_user
        self.end_user = end_user

    def cache_dc(self, end_user, input_data, output_data0, c_d):

        """The  caching processes takes users input data and computation output, then cache them"""
        print("Caching at Data center is done at %d%%" % (random.randint(50, 99)))
        c_kd = end_user * (input_data + output_data0)
        if c_kd < c_d:
            cache_capacity_allocation_dc.append(c_kd)
            rw = RandomWords()
            content_name = rw.random_word()
            cache_content_dc = lfu_cache.Cache()
            cache_content_dc.insert(content_name, output_data0)
            DC_caching_decision_variable.append(1)
            return DC_caching_decision_variable, cache_capacity_allocation_dc
        else:
            cache_capacity_allocation_dc.append(0)
            DC_caching_decision_variable.append(0)
            return DC_caching_decision_variable, cache_capacity_allocation_dc

    def compute_dc(self, end_user, input_data, computation_requirement_user, transm_delay_mec_dc):
        """The  computation processes takes users input and process it"""
        pkd = p_d * (computation_requirement_user / ((end_user - 1) * computation_requirement_user))
        execution_latency = (input_data * computation_requirement_user) / (end_user * pkd)
        print("Computation at Data center is done at %d%%" % (random.randint(50, 99)))
        output_data0 = self.input_data * (110 / 100)
        total_exucution_dc = transm_delay_mec_dc + execution_latency
        execution_latency_dc_vector.append(total_exucution_dc)
        computation_capacity_allocation_dc.append(pkd)
        output_data.append(output_data0)
        DC_caching_decision_variable, cache_capacity_allocation_dc = DataCenter.cache_dc(self, end_user, input_data,
                                                                                         output_data0, c_d)
        DC_computation_decision_variable.append(1)
        return output_data, execution_latency_dc_vector, DC_computation_decision_variable, \
               computation_capacity_allocation_dc, DC_caching_decision_variable, cache_capacity_allocation_dc

#######################################################################################################################


class MEC(object):
    """
    MEC has a limited number of MEC servers to allocate computation and caching resources in parallel
       End-user has to request resources and once got it, it has to start the computation and wait for it to finish.
       The input and output of computation are cached for being reused
    """

    def __init__(self,  number_mec_server,  end_user, input_data, computation_deadline,
                 computation_requirement_user, p_mec, c_mec):
        self.number_mec_server = number_mec_server
        self.p_mec = p_mec
        self.c_mec = c_mec
        self.input_data = input_data
        self.computation_deadline = computation_deadline
        self.computation_requirement_user = computation_requirement_user
        self.end_user = end_user

    def cache(self, end_user, input_data, output_data0, c_mec):
        """The  caching processes takes users input data and computation output, then cache them"""
        print("Caching at MEC server is done at %d%% " % (random.randint(50, 99)))
        cache_mc = lfu_cache.Cache()
        c_km = end_user * (input_data + output_data0)
        MEC_server_maximum_capacity_caching.append(c_mec)
        cache_capacity_allocation_MEC.append(c_km)
        rw = RandomWords()
        content_name = str(rw.random_word())
        content_name_mc.append(content_name)
        return content_name_mc, cache_mc, MEC_server_maximum_capacity_caching, cache_capacity_allocation_MEC

    def compute(self, end_user, input_data, computation_deadline, computation_requirement_user, transm_delay_user_bs):
        """The  computation processes takes users input and process it"""
        global execution_latency_dc_vector  # Global need to be declared in the beginning of the function
        global MEC_m_cache_allocation_variable
        global cache_capacity_allocation_MEC
        global MEC_server_maximum_capacity_caching
        global DC_caching_decision_variable
        global DC_computation_decision_variable
        global output_data

        pkm = p_mec * (computation_requirement_user / ((end_user - 1) * computation_requirement_user))
        if computation_requirement_user >= pkm:
            p_mec_n = max(collaboration_space_capacity)
            number_hop = collaboration_space_capacity.index(p_mec_n)
            pkn = p_mec_n * (computation_requirement_user / ((end_user - 1) * computation_requirement_user))
            execution_latency_mc_n = (input_data * computation_requirement_user) / (end_user * pkn)
            if computation_requirement_user <= pkn and execution_latency_mc_n <= computation_deadline:
                MEC_n_cache_allocation_variable.append(1)
                MEC_m_MEC_n_offloading_variable.append(1)
                MEC_m_cache_allocation_variable.append(0)
                MEC_n_computation_allocation_variable.append(1)
                MEC_m_computation_allocation_variable.append(0)
                # Offloading delay between BS m and BS n
                transm_delay_bsm_bsn = input_data / Bandwidth_BS_m_BS_n
                total_executing_mc_n = transm_delay_user_bs + execution_latency_mc_n + transm_delay_bsm_bsn
                MEC_server_maximum_capacity_computation.append(collaboration_space_capacity[number_hop])
                print("  Computation at MEC is done at %d%%" %(random.randint(50, 99)))
                output_data0 = input_data * (110 / 100)
                content_name_mc, cache_mc, MEC_server_maximum_capacity_caching, cache_capacity_allocation_MEC = MEC.cache(
                    self, end_user, input_data, output_data0, c_mec)
                computation_capacity_allocation_MEC.append(pkn)
                execution_latency_dc_vector.append(0)
                cache_capacity_allocation_dc = []
                output_data.append(output_data0)
                computation_capacity_allocation_dc = []
                DC_computation_decision_variable.append(0)
                computation_capacity_allocation_dc.append(0)
                DC_caching_decision_variable.append(0)
                cache_capacity_allocation_dc.append(0)
                MEC_m_cloud_offloading_variable.append(0)
                total_executing_mc_n_array.append(total_executing_mc_n)
                total_executing_mc_m_array.append(0)
                transm_delay_between_bs.append(transm_delay_bsm_bsn)
                return transm_delay_between_bs, content_name_mc, cache_mc, MEC_m_cloud_offloading_variable, MEC_n_computation_allocation_variable,\
                       MEC_server_maximum_capacity_computation, total_executing_mc_m_array,total_executing_mc_n_array, \
                       computation_capacity_allocation_MEC, cache_capacity_allocation_MEC, MEC_server_maximum_capacity_caching, \
                       MEC_m_cache_allocation_variable,MEC_n_cache_allocation_variable, \
                       MEC_m_MEC_n_offloading_variable,output_data, execution_latency_dc_vector,\
                       DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable,\
                       cache_capacity_allocation_dc
            else:
                print(computation_requirement_user)
                print("No available resources at MEC server, task is offloaded to Data center")
                MEC_n_computation_allocation_variable.append(0)
                MEC_m_computation_allocation_variable.append(0)
                MEC_m_cloud_offloading_variable.append(1)
                MEC_m_cache_allocation_variable.append(0)
                cache_capacity_allocation_MEC.append(0)
                MEC_n_cache_allocation_variable.append(0)
                MEC_m_MEC_n_offloading_variable.append(0)
                transm_delay_mec_dc = np.multiply(1, (input_data / Bandwidth_BS_m_cloud))
                content_name_mc = []
                content_name_mc.append(0)
                cache_mc = []
                cache_mc.append(0)
                # Request Resource at Data center
                datacenter = DataCenter(number_dc_server, computation_deadline, input_data, computation_deadline,
                                    computation_requirement_user, p_d, c_d)
                output_data, execution_latency_dc_vector, DC_computation_decision_variable, computation_capacity_allocation_dc, \
                DC_caching_decision_variable, cache_capacity_allocation_dc = datacenter.compute_dc(end_user,
                                                                               input_data, computation_requirement_user,
                                                                                                   transm_delay_mec_dc)
                MEC_server_maximum_capacity_caching.append(c_mec)
                MEC_server_maximum_capacity_computation.append(self.p_mec)
                computation_capacity_allocation_MEC.append(0)
                total_executing_mc_m_array.append(0)
                total_executing_mc_n_array.append(0)
                transm_delay_bsm_dc = input_data / Bandwidth_BS_m_cloud
                transm_delay_between_bs.append(transm_delay_bsm_dc)
                return transm_delay_between_bs, content_name_mc, cache_mc, MEC_m_cloud_offloading_variable, MEC_n_computation_allocation_variable, \
                       MEC_server_maximum_capacity_computation, total_executing_mc_m_array, total_executing_mc_n_array, \
                       computation_capacity_allocation_MEC, cache_capacity_allocation_MEC, MEC_server_maximum_capacity_caching, \
                       MEC_m_cache_allocation_variable, MEC_n_cache_allocation_variable, \
                       MEC_m_MEC_n_offloading_variable, output_data, execution_latency_dc_vector, \
                       DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
                       cache_capacity_allocation_dc

        else:
            execution_latency_mc = (input_data * computation_requirement_user) / (end_user * pkm)
            total_executing_mc_m = transm_delay_user_bs + execution_latency_mc
            total_executing_mc_m_array.append(total_executing_mc_m)
            total_executing_mc_n_array.append(0)
            MEC_m_MEC_n_offloading_variable.append(0)
            MEC_n_cache_allocation_variable.append(0)
            print("  Computation at MEC is done at %d%%" %
              (random.randint(50, 99)))
            MEC_m_computation_allocation_variable.append(1)
            MEC_n_computation_allocation_variable.append(0)
            MEC_server_maximum_capacity_computation.append(self.p_mec)
            output_data0 = self.input_data * (110 / 100)
            MEC_m_cache_allocation_variable.append(1)
            content_name_mc, cache_mc, MEC_server_maximum_capacity_caching, cache_capacity_allocation_MEC=\
                MEC.cache(self, end_user, input_data, output_data0, c_mec)
            computation_capacity_allocation_MEC.append(pkm)
            execution_latency_dc_vector.append(0)
            cache_capacity_allocation_dc = []
            computation_capacity_allocation_dc = []
            DC_computation_decision_variable.append(0)
            computation_capacity_allocation_dc.append(0)
            DC_caching_decision_variable.append(0)
            cache_capacity_allocation_dc.append(0)
            output_data.append(output_data0)
            transm_delay_between_bs.append(0)
            return transm_delay_between_bs, content_name_mc, cache_mc, MEC_m_cloud_offloading_variable, MEC_n_computation_allocation_variable, \
                   MEC_server_maximum_capacity_computation, total_executing_mc_m_array, total_executing_mc_n_array, \
                   computation_capacity_allocation_MEC, cache_capacity_allocation_MEC, MEC_server_maximum_capacity_caching, \
                   MEC_m_cache_allocation_variable, MEC_n_cache_allocation_variable, \
                   MEC_m_MEC_n_offloading_variable, output_data, execution_latency_dc_vector, \
                   DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
                   cache_capacity_allocation_dc
#######################################################################################################################


"""
Gets user location and gives maximum distance from the base station
"""
random_theta = np.random.uniform(0.0, 1.0, size=120)*2*np.pi
random_radius = ground_distance * np.sqrt(np.random.uniform(0.0, 1.0, size=120))
x = random_radius * np.cos(random_theta)
y = random_radius * np.sin(random_theta)
uniform_distance_points = [(x[i], y[i]) for i in range(120)]


def get_user_location():
    select_location = np.random.randint(10, 120)
    x_uniform=uniform_distance_points[select_location][0]
    y_uniform=uniform_distance_points[select_location][1]
    user_location = math.sqrt(x_uniform**2+y_uniform**2)
    return user_location
#######################################################################################################################
# PathLoss calculates the reference Path Loss between Tx and Rx.


def communication_resources():
    distance_BS_user = get_user_location()
    RSL = 25  # Resource block
    N = end_user    # Number of users
    Interference_level_db = RSL + 10 * math.log10(N - 1) if (N > 1) else 0
    Interference_level_linear = 10 ** (Interference_level_db / 10)
    noise_level_linear = 10 ** (-110 / 10)
    total_noise_linear = noise_level_linear + Interference_level_linear
    total_noise_db = 10 * math.log10(total_noise_linear)
    # Fading
    sigma = 7   # Standard  deviation[dB]
    mu = 0  # Zero mean
    tempx = np.random.normal(mu, sigma, N)
    x= np.mean(tempx)  # In term of dBm
    PL_0_dBm = 34  # In terms of dBm;
    PL_dBm = PL_0_dBm + 10 * path_loss_factor * math.log10(distance_BS_user / ground_distance) + x
    path_loss = 10 ** (PL_dBm / 10)  # [milli - Watts]
    channel_gain = transmission_power - path_loss
    channel_gain = float(channel_gain)
    spectrum_efficiency = math.log(transmission_power * channel_gain ** 2) / total_noise_db ** 2
    return spectrum_efficiency

########################################################################################################################


class EndUser(object):
    def __init__(self, end_user_id, input_data, computation_deadline, computation_requirement_user, p_k, c_k):
        self.p_k = p_k
        self.c_k = c_k
        self.input_data = input_data
        self.computation_deadline = computation_deadline
        self.computation_requirement_user= computation_requirement_user
        self.end_user = end_user_id

    def cache_end_user(self, end_user, input_data, output_data0, c_k):
        """The  caching processes takes users input data and computation output, then cache them"""
        print("computation at end-user device is done at %d%% of %s's task." %
              (random.randint(50, 99), end_user))
        user_cache_allocation_variable.append(1)  # Local caching
        print("Caching at end-user device is done at %d%% of %s's task." %
              (random.randint(50, 99), end_user))
        c_ki = user_cache_allocation_variable * c_k * ((input_data + output_data0) / ((number_application - 1)
                                                                                      * computation_requirement_user))
        rw = RandomWords()
        content_name = rw.random_word()
        content_name_user = []
        content_name_user.append(content_name)
        cache_user = lfu_cache.Cache()
        cache_user.insert(content_name, input_data)
        cache_capacity_allocation_user.append(c_ki)
        return cache_capacity_allocation_user

    def compute_end_user(self, end_user, input_data, computation_deadline, computation_requirement_user):
        """Each end-user demand arrives at BS and requests resources, where each end-user has identification (end_user_id).
          It  starts  computation process, waits for it to finish """
        global total_executing_mc_m_array
        global total_executing_mc_n_array
        global MEC_m_cloud_offloading_variable
        global MEC_server_maximum_capacity_computation
        global MEC_m_computation_allocation_variable
        global MEC_n_computation_allocation_variable
        global output_data
        global transm_delay_between_bs

        # Convert
        # cycle_per_second = cycle_per_byte *  byte_per_second
        # https: // crypto.stackexchange.com / questions / 8405 / how - to - calculate - cycles - per - byte
        # each end-user device has a CPU peak bandwidth of $16$-bit values per cycle
        computation_requirement_user * 16
        pki = p_k * (computation_requirement_user / (end_user * computation_requirement_user))

        execution_latency_user = (input_data * computation_requirement_user) / (1 + pki)
        energy_consumption = cpu_arc_parameter * input_data * computation_requirement_user * pki ** 2
        active_user_list.append(end_user)
        if execution_latency_user >= computation_deadline or energy_consumption >= cpu_energy \
                or computation_requirement_user >= pki:
            print("No available resources at end-user device, task is offloaded to MEC server")
            user_cache_allocation_variable.append(0)
            user_offloading_variable.append(1)
            # Radio resource revenue
            percentage_radio_spectrum = random.random()
            spectrum_efficiency_user = communication_resources()  # End user needs communication resource for offloading
            instantaneous_data = np.multiply(1, (percentage_radio_spectrum *
                                                 spectrum_efficiency_user * channel_bandwidth))
            transm_delay_user_bs = np.multiply(user_offloading_variable, (input_data / 1 + instantaneous_data))
            transm_delay_user_bs = np.amax(transm_delay_user_bs)
            instantaneous_data_vector.append(instantaneous_data)
            input_data0 = []
            input_data0.append(input_data)
            global percentage_radio_spectrum_vector
            percentage_radio_spectrum_vector.append(percentage_radio_spectrum)
            MEC_m_cloud_offloading_variable.append(0)

            # offload to MEC network
            mec = MEC(number_mec_server, end_user, input_data, computation_deadline, computation_requirement_user
                      , p_mec, c_mec)

            transm_delay_between_bs, content_name_mc, cache_mc, MEC_m_cloud_offloading_variable, MEC_n_computation_allocation_variable, \
            MEC_server_maximum_capacity_computation, total_executing_mc_m_array, total_executing_mc_n_array, \
            computation_capacity_allocation_MEC, cache_capacity_allocation_MEC, MEC_server_maximum_capacity_caching, \
            MEC_m_cache_allocation_variable, MEC_n_cache_allocation_variable, \
            MEC_m_MEC_n_offloading_variable, output_data, execution_latency_dc_vector, \
            DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
            cache_capacity_allocation_dc = mec.compute(end_user, input_data, computation_deadline,
                                                       computation_requirement_user, transm_delay_user_bs)

            transm_delay.append(transm_delay_user_bs)
            computation_capacity_allocation_user.append(pki)
            local_computation_cost.append(0)
            user_execution_latency_array.append(0)
            input_data_vector.append(self.input_data)
            return transm_delay_between_bs, transm_delay, percentage_radio_spectrum_vector, instantaneous_data_vector, \
                   user_execution_latency_array, computation_capacity_allocation_user, \
                   local_computation_cost, user_cache_allocation_variable, user_offloading_variable, content_name_mc, \
                   cache_mc, MEC_m_cloud_offloading_variable, MEC_m_computation_allocation_variable, \
                   MEC_n_computation_allocation_variable, MEC_server_maximum_capacity_computation, \
                   total_executing_mc_m_array, total_executing_mc_n_array, computation_capacity_allocation_MEC, \
                   cache_capacity_allocation_MEC, MEC_server_maximum_capacity_caching, MEC_m_cache_allocation_variable, \
                   MEC_n_cache_allocation_variable, \
                   MEC_m_MEC_n_offloading_variable, output_data, execution_latency_dc_vector, \
                   DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
                   cache_capacity_allocation_dc, input_data_vector, active_user_list
        else:
            end_user_device_stutus.append(1)
            user_execution_latency_array.append(execution_latency_user)
            user_cache_allocation_variable.append(1)
            total_executing_mc_m_array.append(0)
            total_executing_mc_n_array.append(0)
            MEC_m_cloud_offloading_variable.append(0)
            percentage_radio_spectrum_vector = []
            MEC_m_computation_allocation_variable.append(0)
            MEC_n_computation_allocation_variable.append(0)
            percentage_radio_spectrum_vector.append(0)
            computation_capacity_allocation_MEC = []
            computation_capacity_allocation_MEC.append(0)
            user_offloading_variable.append(0)  # Local computation
            execution_latency_dc_vector = []
            execution_latency_dc_vector.append(0)
            instantaneous_data_vector.append(0)
            computation_capacity_allocation_user.append(pki)
            local_computation_cost0 = (1 - user_offloading_variable) * end_user_device_stutus
            local_computation_cost.append(local_computation_cost0)
            cache_capacity_allocation_MEC = []
            cache_capacity_allocation_MEC.append(0)
            MEC_m_MEC_n_offloading_variable = []
            MEC_m_MEC_n_offloading_variable.append(0)
            MEC_server_maximum_capacity_computation.append(p_mec)
            output_data0 = self.input_data * (110 / 100)
            output_data.append(output_data0)
            content_name_mc = []
            content_name_mc.append(0)
            cache_mc = []
            transm_delay_between_bs.append(0)
            cache_mc.append(0)
            EndUser.cache_end_user(self, end_user, input_data, output_data0, c_k)
            computation_capacity_allocation_MEC, MEC_server_maximum_capacity_caching, MEC_m_cache_allocation_variable, \
            MEC_n_cache_allocation_variable, \
            DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
            cache_capacity_allocation_dc = 0
            input_data_vector.append(self.input_data)
            transm_delay.append(0)
            return transm_delay_between_bs, transm_delay, percentage_radio_spectrum_vector, instantaneous_data_vector, \
                   user_execution_latency_array, computation_capacity_allocation_user, \
                   local_computation_cost, user_cache_allocation_variable, user_offloading_variable, content_name_mc, \
                   cache_mc, MEC_m_cloud_offloading_variable, MEC_m_computation_allocation_variable, \
                   MEC_n_computation_allocation_variable, MEC_server_maximum_capacity_computation, \
                   total_executing_mc_m_array, total_executing_mc_n_array, computation_capacity_allocation_MEC, \
                   cache_capacity_allocation_MEC, MEC_server_maximum_capacity_caching, MEC_m_cache_allocation_variable, \
                   MEC_n_cache_allocation_variable, \
                   MEC_m_MEC_n_offloading_variable, output_data, execution_latency_dc_vector, \
                   DC_computation_decision_variable, computation_capacity_allocation_dc, DC_caching_decision_variable, \
                   cache_capacity_allocation_dc, input_data_vector, active_user_list

 # Define setup function
########################################################################################################################

# Start the simulation

time_step = 100
for i in range(1, time_step):
    end_user += int(random.expovariate(0.015))  # Generate user with poisson process by using random.expovariate()
    input_data = random.uniform(2, 7)  # input data in terms of Gigabyte (from 2 to 7 GB)
    #  in python
    computation_requirement_user = random.uniform(452.5, 737.5)  # computation requirement for end-users in terms of
    #  cycles/byte
    # example: https://crypto.stackexchange.com/questions/8405/how-to-calculate-cycles-per-byte
    # cycles per byte = cycles per  second/ bytes per second = 2.1 GHz/4.3MBps =
    # (2.1 * 10 power 9)/(4.3 * 1024 power 2) = 466 cpb

    p_k = random.randint(500000000, 1000000000) # Computation capacity at each end-user in terms of Cycles Per Second
                                                # Range from 0.5 GHz to 1.0 GHz

    c_k = random.uniform(10, 64)  # Storage capacity at each end-user device in terms of gigabyte (from 10 to 64)
    users = EndUser(end_user, input_data, computation_deadline, computation_requirement_user, p_k, c_k)

    transm_delay_between_bs, transm_delay, percentage_radio_spectrum_vector, instantaneous_data_vector, \
    user_execution_latency_array, computation_capacity_allocation_user, local_computation_cost, \
    user_cache_allocation_variable, user_offloading_variable,content_name_mc, cache_mc, MEC_m_cloud_offloading_variable,\
    MEC_m_computation_allocation_variable, MEC_n_computation_allocation_variable, MEC_server_maximum_capacity_computation,\
    total_executing_mc_m_array,total_executing_mc_n_array, computation_capacity_allocation_MEC, cache_capacity_allocation_MEC, \
    MEC_server_maximum_capacity_caching, MEC_m_cache_allocation_variable, MEC_n_cache_allocation_variable, \
    MEC_m_MEC_n_offloading_variable, output_data, execution_latency_dc_vector, DC_computation_decision_variable,\
    computation_capacity_allocation_dc, DC_caching_decision_variable, cache_capacity_allocation_dc, input_data_vector, \
    active_user_list = users.compute_end_user(end_user, input_data, computation_deadline, computation_requirement_user)

########################################################################################################################
# Caching analysis
print("communication1", instantaneous_data_vector)
print("Caching1", cache_capacity_allocation_MEC)

g = len(content_name_mc)
for t in range(0, g):
    cache_mc_udate = lfu_cache.Cache()
    cache_mc_udate.insert(content_name_mc[t], output_data[t])

rw = RandomWords()
content_demand = rw.random_words(count=time_step)
content_demand = [x.encode('ascii') for x in content_demand]
T_content_demand = content_demand + content_name_mc
cache_hit_array = []
cache_miss_array = []
N_demand_content = []
data_size = []
min_number_demand = 578
max_number_demand = 3200
for j in range(min_number_demand,  max_number_demand):
    i = 0
    cache_hit = 0
    cache_miss = 0
    hit_size = 0
    while i < j:
        k = int(i/22)
        content_name_find = T_content_demand[k]
        if content_name_find in content_name_mc:
            cache_hit = cache_hit+1
            hit_size = random.uniform(2, 7)
        else:
            cache_miss = cache_miss+1
        i += 1
    cache_hit_array.append(cache_hit)
    cache_miss_array.append(cache_miss)
    data_size.append(hit_size)
    N_demand_content.append(j)
# the probability density function of contents
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.zipf.html
a = 2.0  # parameter
s = np.random.zipf(a, g)
count, bins, ignored = plt.hist(s[s<time_step], time_step, density=True)
x = np.arange(1., time_step)
y = x**(-a) / special.zetac(a)
plt.plot(x, y/max(y), linewidth=3, color='r')
plt.grid(color='gray', linestyle='dashed')
plt.xlabel('100 sample contents', fontsize=18)
plt.ylabel('CDF', fontsize=18)
plt.show

# define zipf distribution parameter
# https://stackoverflow.com/questions/43601074/zipf-distribution-how-do-i-measure-zipf-distribution-using-python-numpy
a0 = 2.0
a1 = 1.0
a2 = 0.5
a3 = 0.1
s = np.trim_zeros(cache_hit_array)
s = np.array(s)
# Display the histogram of the samples, along with the probability density function:
fig2 = plt.figure()
x = np.arange(1., time_step) # We assume that each time slot user offlaod content and cache get cached at MEC server
y0 = x**(-a0) / special.zetac(a0)
y1 = x**(-a1) / special.zetac(a1)
y2 = x**(-a2) / special.zetac(a2)
y3 = x**(-a3) / special.zetac(a3)
a00, = plt.plot(x, y0/max(y0), linewidth=4, color='r', linestyle='--', marker='x')
a11, = plt.plot(x, y1/max(y1), linewidth=4, color='g', linestyle='--', marker='s')
a22, = plt.plot(x, y2/max(y2), linewidth=4, color='b', linestyle='--', marker='+')
a33, = plt.plot(x, y3/max(y3), linewidth=4, color='y', linestyle='--', marker='8')
plt.grid(color='gray', linestyle='dashed')
plt.xlabel('Content index', fontsize=18)
plt.ylabel('Rank of the content', fontsize=18)
plt.legend([a00, a11, a22, a33], ['a = 2.2', 'a = 1.1', 'a = 0.5', 'a = 0.1'], loc='upper right', fancybox=True, fontsize=18)
plt.xlim(1, time_step)
plt.ylim(0, 8)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
fig2.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal3/simulation_4C_02/"
             "plots/content_ranking.pdf", bbox_inches='tight')

# Relationship between caching hit and zipf distribution
# https://en.wikipedia.org/wiki/Zipf%27s_law
m = len(cache_miss_array)
cache_hit_ratio = []
for j in range(0,  m):
    hit_ratio = cache_hit_array[j]/(cache_miss_array[j] + cache_hit_array[j])
    cache_hit_ratio.append(hit_ratio)
zipf_parameter = []
rank_parameter = 0.7
N = len(content_name_mc)
hit_ratio_zipf = []
zipf_cache = []
fig3 = plt.figure()
for i in range(5,  21):
    s = i/10  # i is from o.5 to 2.0
    zipf_parameter.append(s)
    for j in range(1, m):
        hit_ratio2 = cache_hit_array[j] / (cache_miss_array[j] + cache_hit_array[j]*j**rank_parameter)
        hit_ratio_zipf.append(hit_ratio2)
    hit_sum = sum(hit_ratio_zipf)
    num = i-6
    zipf_cache0= 1/(rank_parameter**zipf_parameter[num] * hit_sum)
    zipf_cache.append(zipf_cache0)
    bins = sorted(zipf_cache)
bins=np.round(bins,5)
x_pos = np.arange(len(zipf_parameter))
plt.bar(x_pos, bins, align='center', width=0.7, color='g', alpha=0.5)
plt.plot(x_pos, bins, 'g-', linewidth=3, linestyle='-', marker='^', markersize=10)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(x_pos, zipf_parameter)
plt.xlabel('Zipf exponent parameter (a)', fontsize=18)
plt.ylabel('Normalized cache hit', fontsize=18)
plt.xticks(fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
plt.yticks(fontsize=18)
plt.show()


plt.bar(x_pos, bins, align='center', width=0.7, color='g', alpha=0.5)
plt.plot(x_pos, bins, 'g-', linewidth=3, linestyle='-', marker='^', markersize=10)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(x_pos, zipf_parameter)
plt.xlabel('Zipf exponent parameter (a)', fontsize=18)
plt.ylabel('Normalized cache hit', fontsize=18)
plt.xticks(fontsize=18)
plt.ticklabel_format(style='sci', axis='y')
plt.yticks(fontsize=18)
plt.show()

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'
bins = [x * 100 for x in bins]
plt.bar(x_pos, bins, align='center', width=0.7, color='g', alpha=0.5)
plt.plot(x_pos, bins, 'g-', linewidth=3, linestyle='-', marker='^', markersize=10)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(x_pos, zipf_parameter)
fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')) # No decimal places
#formatter = FuncFormatter(to_percent)
# Set the formatter
plt.gca().yaxis.set_major_formatter(xticks)
plt.xlabel('Zipf exponent parameter (a)', fontsize=18)
plt.ylabel('Cache hit percentage', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

fig3.savefig("C:/Users/anselme/Google Drive/research/Simulation_Research/Journal3/simulation_4C_02/"
             "plots/normalized_cache_hit1.pdf", bbox_inches='tight')



##########################################################################

# BSUM problem

reward1 = [a*b for a, b in zip(MEC_m_computation_allocation_variable, MEC_m_cache_allocation_variable)]
reward2 = [a*b for a, b in zip(MEC_n_computation_allocation_variable, MEC_n_cache_allocation_variable)]
reward = [a+b for a, b in zip(reward2, reward1)]
cache_allocation_variable = [a+b for a, b in zip(MEC_m_cache_allocation_variable, MEC_n_cache_allocation_variable)]
Number_of_request = sum(cache_hit_array)
cache_reward1 = (Number_of_request * output_data)
cache_reward2 = [a*b for a, b in zip(cache_reward1, user_offloading_variable)]
cache_reward = [a*b for a, b in zip(cache_reward2, reward)]

#  local Computation cost
array_one = np.ones(len(user_offloading_variable))
computation_cost10 = np.subtract(array_one, user_offloading_variable)
local_computation_cost_update = [a*b for a, b in zip(computation_cost10, user_execution_latency_array)]
local_computation_cost = [a*b for a, b in zip(local_computation_cost_update, local_computation_cost)]


# Offloading and computation latency
offloading_computation_cost20 = [a*b for a, b in zip(MEC_m_computation_allocation_variable, total_executing_mc_m_array)]
offloading_computation_cost21 = [a*b for a, b in zip(MEC_n_computation_allocation_variable, total_executing_mc_n_array)]
offloading_computation_cost22 = [a*b for a, b in zip(MEC_m_cloud_offloading_variable, execution_latency_dc_vector)]
offloading_computation_cost23 = [a+b for a, b in zip(offloading_computation_cost20, offloading_computation_cost21)]
offloading_computation = [a + b for a, b in zip(offloading_computation_cost22, offloading_computation_cost23)]

# computation delay cost
computation_delay_cost = [a+b for a, b in zip(local_computation_cost, offloading_computation)]
computation = [a * b for a, b in zip(computation_capacity_allocation_MEC, MEC_m_computation_allocation_variable)]
cache_collaboration = [a*b for a, b in zip(reward, output_data)]
p = np.ones(len(user_offloading_variable))
# Optimization problem with BSUM

m = len(user_offloading_variable)
n = number_mec_server
A = np.random.randn(m, n)
epsilon = 1e-6  # Convergence condition
alpha = 12
lamda = 1/np.sqrt(n)
smoothness = 0.01
relaxation_threshold = 7.0
varrho = 1/np.sqrt(m)
L_smooth = (np.linalg.svd(A)[1][0])**2 # L-smooth constant
constant_violation = 1 / np.sqrt(n)
approx_err = []
opt_val_x = []
opt_val_y = []
opt_val_w = []
l = [x * alpha for x in cache_reward]

# Offloading decision variable
x0 = user_offloading_variable
x0 = np.array(x0)

# Computation Decision variable
y0 = [a+b for a, b in zip(MEC_n_computation_allocation_variable, MEC_m_computation_allocation_variable)]
y0 = [a+b for a, b in zip(y0, DC_computation_decision_variable)]
y0 = np.array(y0)


# Caching decision variable
w0 = [a+b for a, b in zip(MEC_m_cache_allocation_variable, MEC_n_cache_allocation_variable)]
w0 = [a+b for a, b in zip(w0, DC_caching_decision_variable)]
w0 = np.array(w0)

#  Objective function
objective_function = abs(np.subtract(computation_delay_cost, l))
objective_function = np.array(objective_function)


def ini_para(opt_x, opt_y, opt_w):
    t = 0  # jth iteration
    # Objective function as defined in equation 34
    opt_val_x.append(objective_function[t] + varrho/2 * np.linalg.norm(np.array(x0) - np.array(opt_x))**2)
    opt_val_y.append(objective_function[t] + varrho/2 * np.linalg.norm(np.array(y0) - np.array(opt_y)) ** 2)
    opt_val_w.append(objective_function[t] + varrho/2 * np.linalg.norm(np.array(w0) - np.array(opt_w)) ** 2)
    # Initial approximate error
    approx_err.append(np.Inf)
    return t, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err

# With CVXPY

########################################################################################################################




# With relaxation
def obj_BSUM(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err, varrho):
    x = cvx.Bool()  # Minimize f over x
    y = cvx.Bool()  # Minimize f over y
    w = cvx.Bool()  # Minimize f over z
    x_xo_norm2 = 0
    y_yo_norm2 = 0
    w_wo_norm2 = 0
    solution_value_relaxation = []
    for t in range(m):
        if t == i:
            x_xo_norm2 += (x - opt_x[t]) ** 2
            y_yo_norm2 += (y - opt_y[t]) ** 2
            w_wo_norm2 += (w - opt_w[t]) ** 2
        else:
            x_xo_norm2 += np.linalg.norm(x0[t] - opt_x[t]) ** 2
            y_yo_norm2 += np.linalg.norm(y0[t] - opt_y[t]) ** 2
            w_wo_norm2 += np.linalg.norm(w0[t] - opt_w[t]) ** 2
        prob1 = cvx.Problem(cvx.Minimize(cvx.sum_entries(objective_function[t] * x + varrho / 2 * x_xo_norm2)),
                            [percentage_radio_spectrum_vector * x <= array_one,
                             computation * x <= MEC_server_maximum_capacity_computation,
                             x * (y0 * (w0 * cache_collaboration)) <= MEC_server_maximum_capacity_caching,
                             MEC_m_computation_allocation_variable <= x, MEC_n_computation_allocation_variable <= x,
                             (array_one - x) + (x * y0) == array_one, MEC_m_cloud_offloading_variable <= x, 0 <= x,
                             x <= 1])
        prob1.solve(verbose=True)
        # Retrieve optimal x value
        opt_x[i]= x.value
        opt_val_x.append(prob1.value)

        # Minimize  over y
        prob2 = cvx.Problem(cvx.Minimize(cvx.sum_entries(objective_function[t] * y + varrho / 2 * y_yo_norm2)),
                            [percentage_radio_spectrum_vector * y <= array_one,
                             computation * y <= MEC_server_maximum_capacity_computation,
                             y * (opt_x * (w0 * cache_collaboration)) <= MEC_server_maximum_capacity_caching,
                             MEC_m_computation_allocation_variable <= y, MEC_n_computation_allocation_variable <= y,
                             (array_one - opt_x) + (opt_x * y) == array_one, MEC_m_cloud_offloading_variable <= y,
                             0 <= y,
                             y <= 1])
        prob2.solve(verbose=True)
        # Retrieve optimal y value
        opt_y[i] = y.value
        opt_val_y.append(prob2.value)

        # Minimize  over w
        prob3 = cvx.Problem(cvx.Minimize(cvx.sum_entries(objective_function[t] * w + varrho / 2 * w_wo_norm2)),
                            [percentage_radio_spectrum_vector * w <= array_one,
                             computation * w <= MEC_server_maximum_capacity_computation,
                             w * (opt_x * (opt_y * cache_collaboration)) <= MEC_server_maximum_capacity_caching,
                             MEC_m_computation_allocation_variable <= w, MEC_n_computation_allocation_variable <= w,
                             (array_one - w) + (w * y0) == array_one, MEC_m_cloud_offloading_variable <= w, 0 <= w,
                             w <= 1])
        prob3.solve(verbose=True)
        # Retrieve optimal w value
        opt_w[i] = w.value
        opt_val_w.append(prob3.value)
        solution_value_relaxation.append(w.value)
        approx_err.append(np.abs((opt_val_w[-2] - opt_val_w[-1]) / opt_val_w[-2]))
        return solution_value_relaxation, opt_val_w, approx_err
########################################################################################################################


# With Rounding Technique
def obj_BSUM_Round(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w,  rounded_decision_value, approx_err, varrho,
                   constant_violation):
    x = cvx.Bool()  # Minimize f over x
    y = cvx.Bool()  # Minimize f over y
    w = cvx.Bool()  # Minimize f over z
    x_xo_norm2 = 0
    y_yo_norm2 = 0
    w_wo_norm2 = 0
    # Maximum violation of communication, computational, and caching capacities
    communication00 = [a * b for a, b in zip(percentage_radio_spectrum_vector, rounded_decision_value)]
    computation00 = [a * b for a, b in zip(computation, rounded_decision_value)]
    caching00 = [a * b for a, b in zip(cache_collaboration, rounded_decision_value)]
    violation_communication0 = [a - b for a, b in zip(communication00, p)]
    violation_computation0 = [a - b for a, b in zip(computation00, MEC_server_maximum_capacity_computation)]
    violation_caching0 = [a - b for a, b in zip(caching00, MEC_server_maximum_capacity_caching)]

    violation_communication = max(0, sum(violation_communication0))
    violation_computation = max(0, sum(violation_computation0))
    violation_caching = max(0, sum(violation_caching0))
    new_max_communication = [violation_communication + b for b in percentage_radio_spectrum_vector]
    new_max_computation = [violation_computation + b for b in MEC_server_maximum_capacity_computation]
    new_max_caching = [violation_caching + b for b in MEC_server_maximum_capacity_caching]
    max_violation = constant_violation * (violation_caching + violation_computation + violation_communication)
    solution_value_rounding = []
    for t in range(m):
        if t == i:
            x_xo_norm2 += (x - opt_x[t]) ** 2
            y_yo_norm2 += (y - opt_y[t]) ** 2
            w_wo_norm2 += (w - opt_w[t]) ** 2
        else:
            x_xo_norm2 += np.linalg.norm(x0[t] - opt_x[t]) ** 2
            y_yo_norm2 += np.linalg.norm(y0[t] - opt_y[t]) ** 2
            w_wo_norm2 += np.linalg.norm(w0[t] - opt_w[t]) ** 2
        prob1 = cvx.Problem(cvx.Minimize(cvx.sum_entries(objective_function[i] * x + varrho / 2 * x_xo_norm2 +
                                                                 max_violation)),
                                    [percentage_radio_spectrum_vector * x <= new_max_communication,
                                     computation * x <= new_max_computation, x * cache_collaboration <= new_max_caching,
                                     MEC_m_computation_allocation_variable <= x,
                                     MEC_n_computation_allocation_variable <= x,
                                     (p - x) + (x * y0) == p, MEC_m_cloud_offloading_variable <= x, 0 <= x, x <= 1])
        prob1.solve(verbose=True)
        # Retrieve optimal x value
        opt_x[i]= x.value
        opt_val_x.append(prob1.value)

        # Minimize  over y
        prob2 = cvx.Problem(cvx.Minimize(cvx.sum_entries(objective_function[i] * y + varrho / 2 * y_yo_norm2 +
                                                                 max_violation)),
                                    [percentage_radio_spectrum_vector * y <= new_max_communication,
                                     computation * y <= new_max_computation, y * cache_collaboration <= new_max_caching,
                                     MEC_m_computation_allocation_variable <= y,
                                     MEC_n_computation_allocation_variable <= y,
                                     (array_one - y) + (y * y0) == array_one, MEC_m_cloud_offloading_variable <= y, 0 <= y, y <= 1])
        prob2.solve(verbose=True)
        # Retrieve optimal y value
        opt_y[i] = y.value
        opt_val_y.append(prob2.value)

        # Minimize  over w
        prob3 = cvx.Problem(cvx.Minimize(cvx.sum_entries(objective_function[i] * w + varrho / 2 * w_wo_norm2 +
                                                                 max_violation)),
                                    [percentage_radio_spectrum_vector * w <= new_max_communication,
                                     computation * w<= new_max_computation, w * cache_collaboration <= new_max_caching,
                                     MEC_m_computation_allocation_variable <= w,
                                     MEC_n_computation_allocation_variable <= w,
                                     (array_one - w) + (w * y0) == array_one, MEC_m_cloud_offloading_variable <= w, 0 <= w, w <= 1])
        prob3.solve(verbose=True)
        # Retrieve optimal w value
        opt_w[i] = w.value
        opt_val_w.append(prob3.value)
        solution_value_rounding.append(w.value)
        approx_err.append(np.abs((opt_val_w[-2] - opt_val_w[-1]) / opt_val_w[-2]))
        return solution_value_rounding, opt_val_w, approx_err
########################################################################################################################

# BSUM with different coordinate selection rules

# Get initial values
opt_x, opt_y, opt_w = np.zeros([m,]), np.zeros([m,]), np.zeros([m,])

t, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err = ini_para(opt_x, opt_y, opt_w)

# BSUM using Cyclic coordinate selection rule
while np.any(approx_err[-1] > epsilon):
    for i in range(0, m):
        solution_value_relaxation_cyc, opt_val_cy, approx_err = obj_BSUM(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w,
                                                                  approx_err, varrho)
        opt_val_cy[t] = opt_val_cy[t] - 1 / 2 * np.gradient(opt_val_cy)[t]
        t += 1
        approx_err_cyc = approx_err
        opt_val_cyc = opt_val_cy

m_relax_cyc = len(solution_value_relaxation_cyc)
opt_x, opt_y, opt_w = np.zeros([m,]), np.zeros([m,]), np.zeros([m,])
t, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err = ini_para(opt_x, opt_y, opt_w)
while np.any(approx_err[-1] > epsilon):
    for i in range(0, m_relax_cyc):
        if solution_value_relaxation_cyc[i] >= relaxation_threshold:
            solution_value_relaxation_cyc[i] = 1
        else:
            solution_value_relaxation_cyc[i] = 0
        solution_value_rounding, opt_Rounding_cyc, approx_err = obj_BSUM_Round(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y,
                                                                         opt_val_w,  solution_value_relaxation_cyc,
                                                                        approx_err, varrho, constant_violation)
        opt_Rounding_cyc[t] = opt_Rounding_cyc[t] - 1 / 2 * np.gradient(opt_Rounding_cyc)[t]
        t += 1
##############################
# BSUM using Gauss-Southwell coordinate selection rule
opt_x, opt_y, opt_w = np.zeros([m,]), np.zeros([m,]), np.zeros([m,])
t, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err = ini_para(opt_x, opt_y, opt_w)
while np.any(approx_err[-1] > epsilon):
    opt_val_prev = opt_val_x
    for i in range(0, m):
        i = np.argmax(np.abs(objective_function + varrho/2 * (np.array(x0) - np.array(opt_x))))
        solution_value_relaxation_gso, opt_val_gou, approx_err = obj_BSUM(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w,
                                                                   approx_err, varrho)
        opt_val_gou[t] = opt_val_gou[t] - 1 / 2 * np.gradient(opt_val_gou)[t]
        t += 1
        approx_err_gso = approx_err
        opt_val_gso = opt_val_gou

m_relax_gso = len(solution_value_relaxation_gso)
opt_x, opt_y, opt_w = np.zeros([m,]), np.zeros([m,]), np.zeros([m,])
t, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err = ini_para(opt_x, opt_y, opt_w)
while np.any(approx_err[-1] > epsilon):
    for i in range(0, m_relax_gso):
        if solution_value_relaxation_gso[i] >= relaxation_threshold:
            solution_value_relaxation_gso[i] = 1
        else:
            solution_value_relaxation_gso[i] = 0
        solution_value_rounding, opt_Rounding_gso, approx_err = obj_BSUM_Round(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y,
                                                                          opt_val_w,  solution_value_relaxation_gso,
                                                                          approx_err, varrho, constant_violation )
        opt_Rounding_gso[t] = opt_Rounding_gso[t] - 1 / 2 * np.gradient(opt_Rounding_gso)[t]
        t += 1
##############################
# BSUM using randomized coordinate selection rule
opt_x, opt_y, opt_w = np.zeros([m,]), np.zeros([m,]), np.zeros([m,])
t, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err = ini_para(opt_x, opt_y, opt_w)
while np.any(approx_err[-1] > epsilon):
    for i in range(0, m):
        i = np.random.randint(0, n)
        solution_value_relaxation_rand, opt_val_rand, approx_err = obj_BSUM(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y,
                                                                     opt_val_w, approx_err, varrho)
        opt_val_rand[t] = opt_val_rand[t] - 1 / 2 * np.gradient(opt_val_rand)[t]
        t += 1
        approx_err_ran = approx_err
        opt_val_ran = opt_val_rand

m_relax_rand = len(solution_value_relaxation_rand)
opt_x, opt_y, opt_w = np.zeros([m,]), np.zeros([m,]), np.zeros([m,])
t, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err = ini_para(opt_x, opt_y, opt_w)
while np.any(approx_err[-1] > epsilon):
    for i in range(0,  m_relax_rand):
        if solution_value_relaxation_rand[i] >= relaxation_threshold:
            solution_value_relaxation_rand[i] = 1
        else:
            solution_value_relaxation_rand[i] = 0
        solution_value_rounding, opt_Rounding_rand, approx_err = obj_BSUM_Round(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y,
                                                                          opt_val_w,  solution_value_relaxation_rand,
                                                                         approx_err, varrho, constant_violation)
        opt_Rounding_rand[t] = opt_Rounding_rand[t] - 1 / 2 * np.gradient(opt_Rounding_rand)[t]
        t += 1

##############################
# ref: Douglas-Rachford splitting and ADMM for nonconvex optimization: tight convergence results
# https://arxiv.org/abs/1709.05747

# Douglas-Rachford threshod


def soft_threshod(w, mu):
    return np.multiply(np.sign(w), np.maximum(np.abs(w) - mu, 0))

def ini_para_DRS():
    t = 0
    # Objective function as defined in equation 34
    opt_val_x.append(objective_function[t])
    opt_val_y.append(objective_function[t])
    opt_val_w.append(objective_function[t])
    return t, opt_val_x, opt_val_y, opt_val_w


def obj_DRS(i, opt_val_x):
    solution_value_relaxation_DRS = []
    x = cvx.Variable()
    computation = [a * b for a, b in zip(computation_capacity_allocation_MEC, MEC_m_computation_allocation_variable)]
    cache_collaboration = [a*b for a, b in zip(reward, output_data)]
    for t in range(m):
        prob_DRS = cvx.Problem(cvx.Minimize(cvx.sum_entries(objective_function[t] * x)),
                           [percentage_radio_spectrum_vector * x <= p,
                            computation * x <= MEC_server_maximum_capacity_computation,
                            x * cache_collaboration <= MEC_server_maximum_capacity_caching,
                            MEC_m_computation_allocation_variable <= x, MEC_n_computation_allocation_variable <= x,
                            (p - x) + (x * y0) == p, MEC_m_cloud_offloading_variable <= x, 0 <= x, x <= 1])
        prob_DRS.solve(verbose=True)
        x0[i] = x.value
        solution_value_relaxation_DRS.append(x.value)
        opt_val_x.append(prob_DRS.value)
        return solution_value_relaxation_DRS, opt_val_x


t, opt_val_x, opt_val_y, opt_val_w = ini_para_DRS()
for i in range(0, m):
    solution_value_relaxation_DRS, opt_valad_DRS = obj_DRS(i, opt_val_x)
    w = opt_valad_DRS
    u = w
    rho = 1/np.sqrt(m)
    w_1 = [x / rho for x in w]

    # Douglas-Rachford threshod
    threshod = soft_threshod(w_1, L_smooth / rho)
    opt_valad_DRS[t] = u[t] + rho * (w[t] - threshod[t])
    t += 1
    obj_DRS_value = opt_valad_DRS[t]

t, opt_val_x, opt_val_y, opt_val_w = ini_para_DRS()
m_relax_DRS = len(solution_value_relaxation_DRS)
for i in range(0, m_relax_DRS):
    if solution_value_relaxation_DRS[i] >= relaxation_threshold:
        solution_value_relaxation_DRS[i] = 1
    else:
        solution_value_relaxation_DRS[i] = 0
    solution_value_rounding, opt_Rounding_DRS, approx_err= obj_BSUM_Round(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y,
                                                                          opt_val_w,  solution_value_relaxation_DRS,
                                                                         approx_err, varrho, constant_violation)
    opt_Rounding_DRS[t] = opt_Rounding_DRS[t] - 1 / 2 * np.gradient(opt_Rounding_DRS)[t]
    t += 1
#############################################
# Visualize DRS-BSUM convergence in terms of  different coordinate selection rules
    fig, ax = plt.subplots(figsize=(9, 6))
    cycDRS_Rounding, = plt.plot(opt_Rounding_cyc, 'r-', linewidth=3, linestyle='--', marker='s')
    gsoDRS_Rounding, = plt.plot(opt_Rounding_gso, 'g-', linewidth=3, linestyle='-', marker='x')
    ranDRS_Rounding, = plt.plot(opt_Rounding_rand, 'b-', linewidth=3, linestyle='--', marker='+')
    DRS_DRS_Rounding, = plt.plot(opt_Rounding_DRS, 'y-', linewidth=3, linestyle='--', marker='^')
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Optimal value of ' r'$\mathcal{B}_j + \xi \Delta$', fontsize=18)
    plt.legend([cycDRS_Rounding, gsoDRS_Rounding, ranDRS_Rounding, DRS_DRS_Rounding],
               ['Cyclic', 'Gauss-Southwell', 'Randomized', 'Douglas-Rachford splitting'], fancybox=True,
               fontsize=18)
    plt.grid(color='gray', linestyle='dashed')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(1, 80)
    plt.ylim(40, 100)
    plt.show()





#############################################


colors22 = ['k', 'm', 'purple', 'c']
cc = itertools.cycle(colors22)
#  Examine how varrho affects the convergence rate
plt.figure()
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Optimal value of ' r'$\mathcal{B}_j$', fontsize=18)


alpha_list = [0.2, 0.5, 10, 100]
alpha_idx = 0
gso = [None] * np.size(alpha_list)
plot_lines = []
for alpha in alpha_list:
    t, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err = ini_para(opt_x, opt_y, opt_w)
    while np.any(approx_err[-1] > epsilon):
        # Pick i-th block, here, 1 block =1 coordinate
        for i in range(0,m):
            i = np.argmax(np.abs(objective_function + varrho/2 * (np.array(x0) - np.array(opt_x))))
            x_value_relaxation_alpha, opt_val_gou, approx_err = obj_BSUM(i, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w,
                                                                 approx_err, varrho)
            opt_val_gou[t] = opt_val_gou[t] - 1 / 2 * np.gradient(opt_val_gou)[t]
            t += 1
    c = next(cc)
    gso[alpha_idx], = plt.plot(opt_val_gou, '-',linestyle='--', linewidth=3, marker='x', label =r'$\varrho$=' + str(alpha), color=c)  # Display results
    alpha_idx += 1
plt.legend(handles=gso)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlim(1, 80)
plt.ylim(40, 100)
cycDRS, = plt.plot(opt_val_cyc, 'r-', linewidth=3, linestyle='--', marker='s' )
gsoDRS, = plt.plot(opt_val_gso, 'g-', linewidth=3, linestyle='-', marker='x')
ranDRS, = plt.plot(opt_val_ran, 'b-', linewidth=3, linestyle='--', marker='+')
DRS_DRS, = plt.plot(obj_DRS_value, 'y-', linewidth=3, linestyle='--', marker='^')
plot_lines.append([cycDRS, gsoDRS, ranDRS, DRS_DRS])
first_legend = plt.legend([cycDRS, gsoDRS, ranDRS, DRS_DRS], ['Cyclic', 'Gauss-Southwell', 'Randomized',
                                                              'Douglas-Rachford splitting'], loc='upper right',
                          fancybox=True, fontsize=18)
# Add the legend manually to the current Axes.
ax = plt.gca().add_artist(first_legend)
plt.legend(handles=gso, loc='center', fancybox=True, fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(1, 80)
plt.ylim(40, 100)
plt.show()

###########################


plt.figure()
plt.xlabel('Iterations', fontsize=18); plt.ylabel('Optimal value of ' r'$\mathcal{B}_j + \xi \Delta$', fontsize=18)
alpha_list = [0.02, 0.1, 0.5, 2]
alpha_idx = 0
gso_violation = [None] * np.size(alpha_list)
for alpha in alpha_list:
    t, opt_x, opt_y, opt_w, opt_val_x, opt_val_y, opt_val_w, approx_err = ini_para(opt_x, opt_y, opt_w)
    for i in range(0, m):
        solution_value_weight, opt_Rounding_wight, approx_err = obj_BSUM_Round(i, opt_x, opt_y, opt_w, opt_val_x,
                                                                               opt_val_y,
                                                                               opt_val_w, solution_value_relaxation_cyc,
                                                                               approx_err, varrho, constant_violation)
        opt_Rounding_wight[t] = opt_Rounding_wight[t] - 1 / 2 * np.gradient(opt_Rounding_wight)[t]
        t += 1
    gso_violation[alpha_idx], = plt.plot(opt_Rounding_wight, '-', label=''r'$\xi$ = ' + str(alpha), linewidth=3) #Display results
    alpha_idx += 1
plt.legend(handles=gso_violation,  fancybox=True, fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(1, 80)
plt.ylim(40, 100)
plt.show()
########################################################################################################################

opt_latency = []
m_1 = len(offloading_computation)
total_transm_latency = [a + b for a, b in zip(transm_delay, transm_delay_between_bs)]
print("total_transm_latency", total_transm_latency)
# offloading_computation =[a + b for a, b in zip(total_transm_latency, offloading_computation)]


def obj_latency(i, off_loc, off_remote):
    x = cvx.Variable()
    prob_latency = cvx.Problem(cvx.Minimize(cvx.sum_entries(((1-x) * off_loc[i]) + (x * off_remote[i]))),
                               [0.02 <= x, x <= 12])
    prob_latency.solve(verbose=True)
    opt_latency.append(prob_latency.value)
    return opt_latency

# Cyclic Coordinate
for i in range(0, m_1):
    latency_value1 = obj_latency(i, local_computation_cost, offloading_computation)
latency_value1 = np.trim_zeros(latency_value1)

# Gauss - Southwell Coordinate
for i in range(0, m_1):
    i = np.argmax(abs(np.gradient(objective_function)))
    latency_value2 = obj_latency(i, local_computation_cost, offloading_computation)
    latency_value2[i] = latency_value2[i] - 1/2 * np.gradient(latency_value2)[i]
latency_value2 = np.trim_zeros(latency_value2)

# Randomized Coordinate
for i in range(0, m_1):
    i = np.random.randint(0, m)
    latency_value3 = obj_latency(i, local_computation_cost, offloading_computation)
    latency_value3[i] = latency_value3[i] - 1/2 * np.gradient(latency_value3)[i]
latency_value3 = np.trim_zeros(latency_value3)

# Douglas Rachford spliting
for i in range(0, m_1):
    latency_value4 = obj_latency(i, local_computation_cost, offloading_computation)
    w1 = latency_value4
    u1 = w1
    rho1 = 2
    w_11 = [x / rho1 for x in w1]

    # Douglas-Rachford threshod
    threshod_latency = soft_threshod(w_11, L_smooth / rho1)
    latency_value4[i] = latency_value4[i] + rho1 * (w1[i]-threshod_latency[i])
latency_value4 = np.trim_zeros(latency_value4)

raw_data_delay = {'Cyc': latency_value1, 'G-S': latency_value2, 'Ran': latency_value3, 'D-R-S': latency_value4}
df_delay = pd.DataFrame.from_dict(raw_data_delay, orient='index')
df_delay = df_delay.transpose()
fig, ax = plt.subplots()
medianprops = dict(linestyle='-', linewidth=4, color='blue')
bp = df_delay.boxplot(column=['Cyc', 'G-S', 'Ran', 'D-R-S'],  showbox=True, notch=True, patch_artist=True,
                      showmeans=True, meanline=True,medianprops=medianprops, showfliers=False, return_type='dict')
plt.ylim(0, 0.3)
plt.ylabel('Computation latency', fontsize=18)

for patch in bp['boxes']:
        patch.set(facecolor='yellowgreen',linewidth=3)

## change color and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#228b22', linewidth=3)

## change color and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#228b22', linewidth=3)

## change color and linewidth of the medians
for median in bp['medians']:
    median.set(color='blue', linewidth=3)
## change color and linewidth of the medians
for median in bp['means']:
    median.set(color='black', linewidth=3)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o', color='yellow', alpha=0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
plt.grid(color='gray', linestyle='dashed')
plt.show()


############
opt_transm_delay = []
m_delay = len(transm_delay)


def transm_obj_latency(i, total_transm_latency):
    x = cvx.Variable()
    # Transmission time must be less than compuatation deadline, otherwise we  can not offload
    prob_transm_latency = cvx.Problem(cvx.Minimize(cvx.sum_entries(total_transm_latency[i] * x)),
                                      [0.02 <= x, x <= 12])
    prob_transm_latency.solve(verbose=True)
    opt_transm_delay.append(prob_transm_latency.value)
    return opt_transm_delay

# Cyclic Coordinate
for i in range(0, m_delay):
    transm_latency_value1 = transm_obj_latency(i, total_transm_latency)
transm_latency_value1 = np.trim_zeros(transm_latency_value1)

# Gauss - Southwell Coordinate
for i in range(0, m_delay):
    i = np.argmax(abs(np.gradient(total_transm_latency)))
    transm_latency_value2 = transm_obj_latency(i, total_transm_latency)
transm_latency_value2 = np.trim_zeros(transm_latency_value2)

# Randomized Coordinate
for i in range(0, m_delay):
    i = np.random.randint(0, m_delay)
    transm_latency_value3 = transm_obj_latency(i, total_transm_latency)
transm_latency_value3 = np.trim_zeros(transm_latency_value3)


# Douglas Rachford spliting
for i in range(0, m_delay):
    transm_latency_value4 = transm_obj_latency(i, total_transm_latency)
    w1 = transm_latency_value4
    u1 = w1
    rho1 = 2
    w_11 = [x / rho1 for x in w1]

    # Douglas-Rachford threshod
    transm_threshod_latency4 = soft_threshod(w_11, L_smooth / rho1)
transm_latency_value4 = np.trim_zeros(transm_latency_value4)

raw_data_transm_delay = {'Cyc': transm_latency_value1, 'G-S': transm_latency_value2, 'Ran': transm_latency_value3, 'D-R-S': transm_latency_value4}
df_transm_delay = pd.DataFrame.from_dict(raw_data_transm_delay, orient='index')
df_transm_delay = df_transm_delay.transpose()
fig2, ax = plt.subplots()
medianprops = dict(linestyle='-', linewidth=4, color='blue')
bp2 = df_transm_delay.boxplot(column=['Cyc', 'G-S', 'Ran', 'D-R-S'],  showbox=True, notch=True, patch_artist=True,
                      showmeans=True, meanline=True,medianprops=medianprops, showfliers=False, return_type='dict')

plt.ylabel('Transmission latency', fontsize=18)

for patch in bp2['boxes']:
        patch.set(facecolor='yellowgreen',linewidth=3)
for whisker in bp2['whiskers']:
    whisker.set(color='#228b22', linewidth=3)

## change color and linewidth of the caps
for cap in bp2['caps']:
    cap.set(color='#228b22', linewidth=3)

## change color and linewidth of the medians
for median in bp2['medians']:
    median.set(color='blue', linewidth=3)
## change color and linewidth of the medians
for median in bp2['means']:
    median.set(color='black', linewidth=3)

## change the style of fliers and their fill
for flier in bp2['fliers']:
    flier.set(marker='o', color='yello', alpha=0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(-1, 1))
plt.grid(color='gray', linestyle='dashed')
plt.show()
########################################################################################################################

# Analyzing bandwidth saving

total_bandwidth_saving = [a * b for a, b in zip(data_size, cache_hit_array)]
total_bandwidth_saving = [a * b for a, b in zip(total_bandwidth_saving, N_demand_content)]
# Zero are cache miss, while one is cache hit. we remove cache misses becouse they do not contribute to bandwidth saving
total_bandwidth_saving = np.trim_zeros(total_bandwidth_saving)
weight_parameter = [0.5, 0.7, 1.0]
weighted_bandwidth_saving1 = [x * weight_parameter[0] for x in total_bandwidth_saving]
weighted_bandwidth_saving2 = [x * weight_parameter[1] for x in total_bandwidth_saving]
weighted_bandwidth_saving3 = [x * weight_parameter[2] for x in total_bandwidth_saving]
m_2 = len(total_bandwidth_saving)

opt_bandwidth = []


def obj_bandwidth(i, weighted_bandwidth_saving):
    x = cvx.Variable()
    prob_bandwidth = cvx.Problem(cvx.Maximize(cvx.sum_entries(x * weighted_bandwidth_saving[i])),
                                 [0 <= x, x <= MEC_server_maximum_capacity_caching])
    prob_bandwidth.solve(verbose=True)
    opt_bandwidth.append(prob_bandwidth.value)
    return opt_bandwidth


weighted_bandwidth_saving1=sorted(weighted_bandwidth_saving1)
Y1 = []
l1 = len(weighted_bandwidth_saving1)
Y1.append(float(1)/l1)
for i1 in range(2, l1+1):
    Y1.append(float(1)/l1+Y1[i1-2])
weighted_bandwidth_saving2=sorted(weighted_bandwidth_saving2)
Y2 = []
l2 = len(weighted_bandwidth_saving2)
Y2.append(float(1)/l2)
for i2 in range(2, l2+1):
    Y2.append(float(1)/l2+Y2[i2-2])
weighted_bandwidth_saving3=sorted(weighted_bandwidth_saving3)
Y3 = []
l3 = len(weighted_bandwidth_saving3)
Y3.append(float(1)/l3)
for i3 in range(2,l3+1):
    Y3.append(float(1)/l3+Y3[i3-2])


bandwidth1, = plt.plot(weighted_bandwidth_saving1, Y1, 'r-', linewidth=3, linestyle='--', marker='s')
bandwidth2, = plt.plot(weighted_bandwidth_saving2, Y2, 'g-', linewidth=3, linestyle='--', marker='x')
bandwidth3, = plt.plot(weighted_bandwidth_saving3, Y3, 'b-', linewidth=3, linestyle='--', marker='+')
plt.xlabel('Bandwidth saving (GB)', fontsize=18)
plt.ylabel('CDF', fontsize=18)
plt.legend([bandwidth1, bandwidth2, bandwidth3], ['' r'$\eta$ = 0.5', '' r'$\eta$ = 0.7', '' r'$\eta$ = 1.0'],
           fancybox=True, fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


########################################################################################################################
# Plot Resources allocation
# Computation resource
Convert_rate = 11822
opt_compute = []
m_comput = len(computation_capacity_allocation_MEC)


def obj_compute(i, computation_allocation):
    x = cvx.Variable()
    prob_compute = cvx.Problem(cvx.Minimize(cvx.sum_entries(x * computation_allocation[i])),
                               [1855250 <= x, x <= 3023750])
    prob_compute.solve(verbose=True)
    opt_compute.append(prob_compute.value)
    return opt_compute

# Cyclic Coordinate


computation_capacity_allocation_MEC = [x / (Convert_rate * end_user) for x in computation_capacity_allocation_MEC]

for i in range(0, m_comput):
    compute_value1 = obj_compute(i, computation_capacity_allocation_MEC)
    compute_value1[i] = abs(compute_value1[i])
    compute_value1 = compute_value1

# Gauss - Southwell Coordinate
for i in range(0, m_comput):
    i = np.argmax(abs(np.gradient(objective_function)))
    compute_value2 = obj_compute(i, computation_capacity_allocation_MEC)
    compute_value2[i] = abs(compute_value2[i] - 1 / 2 * np.gradient(compute_value2)[i])
    compute_value2 = abs(compute_value2 - 1 / 2 * np.gradient(compute_value2)[i])



# Randomized Coordinate
for i in range(0, m_comput):
    i = np.random.randint(0, m_comput)
    compute_value3 = obj_compute(i, computation_capacity_allocation_MEC)
    compute_value3[i] = abs(compute_value3[i] - 1 / 2 * np.gradient(compute_value3)[i])
    compute_value3 = abs(compute_value3 - 1 / 2 * np.gradient(compute_value3)[i])

# Douglas Rachford spliting

for i in range(0, m_1):
    compute_value4 = obj_compute(i, computation_capacity_allocation_MEC)
    w3 = compute_value4
    u3 = w3
    rho3 = 2
    w_3 = [x / rho3 for x in w3]

    # Douglas-Rachford threshod
    threshod_compute = soft_threshod(w_3, L_smooth / rho3)
    compute_value4[i] = abs(compute_value4[i] + rho3 * (w3[i]-threshod_compute[i]))
    compute_value4 = abs(compute_value4 - 1 / 2 * np.gradient(compute_value4)[i])

# Visualize computation resource allocation
# Convert  cycle per second to MIPS,  where 1 MIPS= 11821 cycle per second
#Convert_rate = 11821
X1 = sorted(compute_value1)
#X1 = [x / Convert_rate for x in X1]
Y1 = []
l1 = len(X1)
Y1.append(float(1)/l1)
for i1 in range(2, l1+1):
    Y1.append(float(1)/l1+Y1[i1-2])

X2 = sorted(compute_value2)
#X2 = [x / Convert_rate for x in X2]
Y2 = []
l2 = len(X2)
Y2.append(float(1)/l2)
for i2 in range(2, l2+1):
    Y2.append(float(1)/l2+Y2[i2-2])

X3 = sorted(compute_value3)
#X3 = [x / Convert_rate for x in X3]
Y3 = []
l3 = len(X3)
Y3.append(float(1)/l3)
for i3 in range(2,l3+1):
    Y3.append(float(1)/l3+Y3[i3-2])

X4 = sorted(compute_value4)
#X4 = [x / Convert_rate for x in X4]
Y4 = []
l4 = len(X4)
Y4.append(float(1)/l4)
for i4 in range(2, l4+1):
    Y4.append(float(1)/l4+Y4[i4-2])

cyc2, = plt.plot(X1, Y1, 'r-', linewidth=3, linestyle='--', marker='s')
gso2, = plt.plot(X2, Y2, 'g-', linewidth=3, linestyle='--', marker='x')
ran2, = plt.plot(X3, Y3, 'b-', linewidth=3, linestyle='--', marker='+')
DRS_compute, = plt.plot(X4, Y4, 'y-', linewidth=3, linestyle='--', marker='^')
plt.xlabel('Computation throughput (MIPS)', fontsize=18)
plt.ylabel('CDF', fontsize=18)
plt.legend([cyc2, gso2, ran2, DRS_compute], ['Cyclic', 'Gauss-Southwell', 'Randomized','Douglas-Rachford splitting']
           , fancybox=True, fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=18)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.yticks(fontsize=18)
plt.show()


########################################################################################################################
# Caching resource
m = len(user_offloading_variable)
n = number_mec_server
A = np.random.randn(m, n)
epsilon = 1e-6  # Convergence condition
alpha = 12
lamda = 1/np.sqrt(n)
smoothness = 0.01
relaxation_threshold = 7.0
varrho = 1/np.sqrt(m)
L_smooth = (np.linalg.svd(A)[1][0])**2 # L-smooth constant
constant_violation = 1 / np.sqrt(n)
opt_cache = []
cache_capacity_allocation_MEC = np.trim_zeros(cache_capacity_allocation_MEC)
m_cache = len(cache_capacity_allocation_MEC)


def ini_para_cache():
    opt_cache.append(cache_capacity_allocation_MEC[1])
    return opt_cache


def obj_cache(i, opt_cache, cache_allocation):
    x = cvx.Variable()
    prob_cache = cvx.Problem(cvx.Maximize(cvx.sum_entries(x * cache_allocation[i])), [0 <= x, x <= 500000])
    prob_cache.solve(verbose=True)
    opt_cache.append(prob_cache.value)
    return opt_cache


opt_cache = ini_para_cache()

iteration = []

for i in range(0, m_cache):
    iteration.append(i)
    cache_value = obj_cache(i, opt_cache, cache_capacity_allocation_MEC)
    cache_value[i] = abs(cache_value[i] - 1/2 * np.gradient(cache_value)[i])

cache_value = cache_value[:-1]
cache_value = sorted(cache_value)
iteration = np.array(iteration)
cache_value = np.array(cache_value)
iteration_smooth = np.linspace(iteration.min(), iteration.max(), 1000) # 300: Number of paints in graph
print("iteration", len(iteration))
print("cache_value", len(cache_value))
print("iteration_smooth", len(iteration_smooth))
cache_smooth = spline(iteration, cache_value,iteration_smooth)

# Visualize caching resource allocation
fig, ax = plt.subplots(figsize=(9, 6))
plt.plot(iteration_smooth, cache_smooth, color='yellowgreen', linestyle='--', marker='x')
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Cache resource (GB)', fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 90)
plt.show()

########################################################################################################################
# Communication communication
# How do you convert MHz to Mbps? source: https://onlineconversion.vbulletin.net/forum/main-forums/
# convert-and-calculate/301-conversion-of-bits-mhz-to-mbps
# (X bits * Y Mhz) / 8 = Z Mbps
instantaneous_data_vector = [x * 8 for x in instantaneous_data_vector]
m_com = len(instantaneous_data_vector)
opt_com = []


def ini_para_com():
    opt_com.append(instantaneous_data_vector[1])
    return opt_com


def obj_com(i, opt_com, com_allocation):
    x = cvx.Variable()
    prob_com = cvx.Problem(cvx.Maximize(cvx.sum_entries(x * com_allocation[i])), [0 <= x, x <= 25])
    prob_com.solve(verbose=True)
    opt_com.append(prob_com.value)
    return opt_com


opt_com = ini_para_com()

for i in range(0, m_com):
    com_value = obj_com(i, opt_com, instantaneous_data_vector)
    com_value[i] = abs(com_value[i] - 1 / 2 * np.gradient(com_value)[i])
com_value = com_value[:-1]
com_value = sorted(com_value)
com_value = np.array(com_value)

com_value = np.array(com_value)
com_smooth = spline(iteration, com_value,iteration_smooth)

# Visualize caching resource allocation
fig, ax = plt.subplots(figsize=(9, 6))
plt.plot(iteration_smooth, com_smooth, color='yellowgreen', linestyle='--', marker='x')
plt.xlabel('Iterations', fontsize=18)
plt.ylabel('Network throughput (Mbps)', fontsize=18)
plt.grid(color='gray', linestyle='dashed')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(1, 30)
plt.xlim(1, 90)
plt.show()

#############################################

integrality_gap = min(opt_val_cyc)/min(opt_Rounding_cyc)
print("Integrality_gap"), integrality_gap

endTime = time.time()
#############################################
simulation_time = endTime - startTime
print "Simulation time:", simulation_time, "seconds"
