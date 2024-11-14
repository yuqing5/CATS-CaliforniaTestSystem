#########################################################################################################
# run_opf.jl
#
# Run a DC power flow
#
#########################################################################################################

###########
## Setup ##
###########

# Load Julia Packages
#--------------------
using PowerModels
using JuMP
using CSV, JSON
using DataFrames
using Ipopt
using Memento
# User input
#-----------
save_to_JSON = true

# Specify solver
#----------------
# const IPOPT_ENV = Ipopt.Env()
#Memento.setlevel!(getlogger("PowerModels"), "warn")
#Memento.setlevel!(getlogger("JuMP"), "notice")
#Memento.setlevel!(getlogger("Ipopt"), "notice")
# PowerModels.silence()
solver = JuMP.optimizer_with_attributes(() -> Ipopt.Optimizer(), "print_level" => 0) #Ipopt.Optimizer JuMP.optimizer_with_attributes(() -> Ipopt.Optimizer(), "print_level" => 1)
# Load Data
# ---------
# Load the MATPOWER data file
data = PowerModels.parse_file("CaliforniaTestSystem.m")
###########
## Solve ##
###########

solution = PowerModels.solve_opf(data, DCPPowerModel, solver)
# Save solution dictionary to JSON
if save_to_JSON == true
    stringdata = JSON.json(solution)
    
    # write the file with the stringdata variable information
    open("pf_solution.json", "w") do f
        write(f, stringdata)
    end
end