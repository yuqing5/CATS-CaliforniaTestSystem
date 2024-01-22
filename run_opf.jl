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

# User input
#-----------
save_to_JSON = true

# Specify solver
#----------------
# const IPOPT_ENV = Ipopt.Env()
solver = Ipopt.Optimizer # JuMP.optimizer_with_attributes(() -> Ipopt.Optimizer(), "print_level" => 1)

# Load Data
# ---------
# Load the MATPOWER data file
data = PowerModels.parse_file("MATPOWER/CaliforniaTestSystem.m")

###########
## Solve ##
###########

solution = PowerModels.solve_opf(data, ACPPowerModel, solver)

# Save solution dictionary to JSON
if save_to_JSON == true
    stringdata = JSON.json(solution)
    
    # write the file with the stringdata variable information
    open("pf_solution_ac.json", "w") do f
        write(f, stringdata)
    end
end