import numpy as np
import cvxpy as cp
import random
import csv
from datetime import datetime
import time
import sys
import pandas as pd
import json
import collections
import scipy.io
import mpu
type_to_emission = collections.defaultdict(float)
type_to_emission['Conventional Hydroelectric'] = 0
type_to_emission['Hydroelectric Pumped Storage'] = 0
type_to_emission['Petroleum Liquids'] = 1808.5
type_to_emission['Natural Gas Internal Combustion Engine'] = 1255
type_to_emission['Natural Gas Fired Combined Cycle'] = 1255
type_to_emission['Natural Gas Steam Turbine'] = 1255
type_to_emission['Natural Gas Fired Combustion Turbine'] = 1255
type_to_emission['Nuclear'] = 0
type_to_emission['Geothermal'] = 166.7
type_to_emission['Onshore Wind Turbine'] = 0
type_to_emission['Other Waste Biomass'] = 76
type_to_emission['Wood/Wood Waste Biomass'] = 76
type_to_emission['Landfill Gas'] = 1255
type_to_emission['Solar Photovoltaic'] = 145
type_to_emission['Solar Thermal without Energy Storage'] = 145
type_to_emission['Conventional Steam Coal'] = 1149
type_to_emission['Other Gases'] = 1255
type_to_emission['Batteries'] = 0
type_to_emission['Petroleum Coke'] = 1808.5
type_to_emission['Municipal Solid Waste'] = 884
type_to_emission['Other Natural Gas'] = 1255
type_to_emission['IMPORT'] = 884
type_to_emission['Synchronous Condenser'] = 884
np.random.seed(29)

def recurse_son(father, Spanning_tree, ratio_now, visited, gen_line_prop_mat):
    node_current = father
    visited[node_current] = True
    child_bus, child_connected_line = find_child(A_mat_directed, node_current)
    father_ratio=np.copy(ratio_now)
    #print("Current father", node_current)
    #print("Current son", child_bus)
    #print("Child connected line", child_connected_line)

    if child_bus==[]:
        return

    for i in range(len(child_bus)):
        current_son = child_bus[i]
        if visited[current_son]:
            #print("Already visited, working on son ", current_son)
            #print("Current father ", node_current)
            #print("Current ratio", father_ratio)
            ratio_now = line_prop_mat[child_connected_line[i], node_current] * father_ratio
            gen_line_prop_mat[child_connected_line[i], current_son] = ratio_now
            Spanning_tree[current_son] += bus_prop_vec[current_son] * ratio_now
            #print("New Ratio", ratio_now)
            recurse_son(current_son, Spanning_tree, ratio_now, visited, gen_line_prop_mat)

        if not visited[current_son]:
            #print("Working on son ", current_son)
            #print("Previous ratio", father_ratio)
            ratio_now = line_prop_mat[child_connected_line[i], node_current] * father_ratio
            gen_line_prop_mat[child_connected_line[i], current_son] = ratio_now
            Spanning_tree[current_son] += bus_prop_vec[current_son] * ratio_now
            #print("New Ratio", ratio_now)
            recurse_son(current_son, Spanning_tree, ratio_now, visited, gen_line_prop_mat)
    return



def find_child(directed_graph, current_vertex):
    son=[]
    line_index=[]
    for i in range(num_lines):
        if directed_graph[i][current_vertex]==-1:
            index=np.argwhere(directed_graph[i,:]==1)
            son.append(index)
            line_index.append(i)
    son = np.array(son).reshape(-1, 1)
    line_index=np.array(line_index).reshape(-1,1)
    return son, line_index


np.set_printoptions(threshold=sys.maxsize)
num_gen=2149
num_buses=8870
num_lines=10823

branch_ = scipy.io.loadmat('MATPOWER/branch.mat')['brach']
connection_all = branch_[:, 0:2]

start = time.time()
A_mat = np.zeros((num_lines, num_buses), dtype=float)
for i in range(num_lines):
    A_mat[i][int(connection_all[i][0])-1] = -1.0
    A_mat[i][int(connection_all[i][1])-1] = 1.0

bus = scipy.io.loadmat('MATPOWER/bus.mat')['bus']
load = bus[:, 2]
df = pd.read_csv("GIS/CATS_gens.csv")
df = df[df['Pmax'] != 0.0].to_numpy()
f = open("pf_solution.json")
sol = json.load(f)
Gen_node = [0]*2149
C = [0]*2149
power_generation = [0]*2149
carbon_emissions = [0]*2149
branch_power_to = [0]*10823
branch_power_from = [0]*10823
for line, val in sol['solution']['gen'].items():
    if val['pg'] != 0.0:
        Gen_node[int(line)-1] = df[int(line)-1][2]-1
        carbon_emissions[int(line)-1] = type_to_emission[df[int(line)-1][3]]
        C[int(line)-1] = val['pg_cost']*100
        power_generation[int(line)-1] = val['pg']*100

for line, val in sol['solution']['branch'].items():
    branch_power_from[int(line)-1] = val['pf']*100
    branch_power_to[int(line)-1] = val['pt']*100

f.close()
carbon_emissions = np.array(carbon_emissions)
Gen_node=np.array(Gen_node)
C = np.array(C)
x = np.array(power_generation)
line_flow = np.array(branch_power_to)
# D1= np.diag(np.full(num_gen, 1))
# D=np.stack((D1, -D1), axis=0)
# D=np.reshape(D, (12,6))
# e = np.array([10.0, 15.0, 15.0, 15.0, 15.0, 15.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0])

# b = np.zeros((30, 6), dtype=float)
# b[0][0]=1.0
# b[1][1]=1.0
# b[12][2]=1.0
# b[21][3]=1.0
# b[22][4]=1.0
# b[26][5]=1.0

# x = cp.Variable(num_gen)
# line_flow = cp.Variable(num_lines)

# line_flow_limit = np.ones((1, num_lines), dtype=float) * 4.0
# neg_line_flow_limit = -np.ones((1, num_lines), dtype=float) * 4.0

# cost = cp.sum(C @ x)
# constraint = [b @ x + A_mat.T @ line_flow == load.reshape(-1, ),
#               D @ x <= e,  # Generation constraint
#               line_flow <= line_flow_limit.reshape(-1, ),
#               line_flow >= neg_line_flow_limit.reshape(-1,)]  # line flow limits
# prob = cp.Problem(cp.Minimize(cost), constraint)
# prob.solve(solver=cp.CVXOPT)

#print("Load, ", load.T)
#print("Generation value", np.round(x,2))
#print("Line flow value", np.round(line_flow,3))

A_mat_directed=np.copy(A_mat)

for i in range(num_lines):
    if line_flow[i]<0:
        A_mat_directed[i][int(connection_all[i][0]) - 1] = 1.0
        A_mat_directed[i][int(connection_all[i][1]) - 1] = -1.0

#print("Original graph", A_mat)
#print("Directed graph", A_mat_directed)


line_prop_mat=np.zeros((num_lines, num_buses), dtype=float)
bus_prop_vec=np.zeros((num_buses, 1), dtype=float)

for i in range(num_buses):
    total_power = np.copy(load[i])
    total_inflow = 0.0
    for j in range(num_lines):
        if A_mat_directed[j][i] == -1:
            total_power += np.abs(line_flow[j])
        elif A_mat_directed[j][i] == 1:
            total_inflow += np.abs(line_flow[j])

    #print("Node ", i)
    #print("Total power outflow and demand:", total_power)
    #print("Total line flow injections at this node:", total_inflow)
    bus_prop_vec[i] = load[i] / total_power
    for j in range(num_lines):
        if A_mat_directed[j][i] == -1:
            line_prop_mat[j, i] = np.abs(line_flow[j]) / total_power

print("FINISH!!!!!!!!!")
#print("Directed incidence matrix", A_mat_directed, flush=True)
#print("Line proportion matrix", line_prop_mat)
#print("Bus proportion vector", bus_prop_vec.T)

Gen_prop_mat=np.zeros((num_gen, num_lines, num_buses), dtype=float)
Gen_spanning_tree_all=np.zeros((num_buses, num_gen))

for i in range(num_gen):
    #print("Generator", Gen_node[i])
    #print("Generation Value", x[i])
    Gen_spanning_tree = np.zeros((num_buses, 1), dtype=float)
    current_Gen_prop_mat = np.zeros((num_lines, num_buses), dtype=float)
    visited = np.zeros(num_buses, dtype=bool)
    child = True
    current_ratio = 1.0
    current_node = Gen_node[i]
    Gen_spanning_tree[current_node] = bus_prop_vec[current_node] * current_ratio
    recurse_son(current_node, Gen_spanning_tree, current_ratio, visited, current_Gen_prop_mat)
    #print("Generator's contribution to each load", np.round(Gen_spanning_tree, 3).T)
    #print("Sum of generator's output", np.sum(Gen_spanning_tree))
    #print("Generator's contribution to each line:", current_Gen_prop_mat)
    Gen_prop_mat[i]=current_Gen_prop_mat
    Gen_spanning_tree_all[:, i] = Gen_spanning_tree.reshape(-1, )

load_vec = np.zeros((num_buses, 1), dtype=float)
carbon_vec = np.zeros((num_buses, 1), dtype=float)
for i in range(num_buses):
    for j in range(num_gen):
        load_vec[i] += Gen_spanning_tree_all[i,j] * x[j]
        carbon_vec[i] += Gen_spanning_tree_all[i,j] * x[j] * carbon_emissions[j]
#print("Recovered Load vector", load_vec.T)
#print("Original load vector", load.T)
print("Carbon emissions vector", carbon_vec.T)
average_emission_rate = carbon_vec/load_vec
print("Carbon emissions rate vector", average_emission_rate.T)
end = time.time()
print(end-start)















#node_prop_mat=np.zeros((), dtype=float)