#######################################################################
# File       : dc_syn.tcl
# Description: Example Verilog Synthesis Script (based on Design Compiler)
# Note       : Please modify the script accordingly for your environment,
#              library paths, constraints, etc.
#######################################################################

#======================================================================
# 1. Environment and Variable Settings
#======================================================================
# Define the top-level module name
set top_name    "AxCore_SharedAdd_MPWq4_PE"

# CHECK the clock period
set clk_period 1.0

# Define the path to the standard cell library
set std_cell_lib        "/hpc/Tech/TSMC/digital/N28/digital/Front_End/timing_power_noise/CCS/tcbn28hpcplusbwp30p140_180a/tcbn28hpcplusbwp30p140ffg0p99v0c_ccs.db"

# If there are more libraries (IO library, clock gating library, etc.),
# you can define them here as well
# set io_lib         "/path/to/your/io_db.db"
# set cgs_lib        "/path/to/your/clock_gating_db.db"

# Specify the list of root directories to search
set search_dirs {
    "/hpc/home/connect.ychen433/CodeArea/TestAE/AreaTest/PE/W4/MPWq4_E8M23"
}

# Define a recursive function to find Verilog files in a directory and its subdirectories
proc find_verilog_files {dir file_list} {
    upvar $file_list file_list_real ; # Link to the outer variable

    # Get all files and subdirectories in the current directory
    foreach item [glob -directory $dir *] {
        if {[file isdirectory $item]} {
            # If it's a directory, call the function recursively
            find_verilog_files $item file_list_real
        } elseif {[string match *.v $item]} {
            # If it's a Verilog file, add it to the file list
            lappend file_list_real $item
        }
    }
}

# Initialize a list to store found Verilog files
set verilog_list {}

# Iterate over each directory in the list and call the recursive function
foreach search_dir $search_dirs {
    find_verilog_files $search_dir verilog_list
}

# Define working directory
set work_dir "./report/$top_name"
file mkdir $work_dir

# Define output files (netlist, constraints, reports, etc.)
set netlist_out        "${work_dir}/${top_name}_synth.v"
set sdc_out            "${work_dir}/${top_name}_synth.sdc"
set mapped_ddc_out     "${work_dir}/${top_name}_mapped.ddc"
set timing_rpt         "${work_dir}/${top_name}_timing.rpt"
set area_rpt           "${work_dir}/${top_name}_area.rpt"
set power_rpt          "${work_dir}/${top_name}_power.rpt"

file mkdir [file dirname $netlist_out]
file mkdir [file dirname $sdc_out]
file mkdir [file dirname $mapped_ddc_out]
file mkdir [file dirname $timing_rpt]
file mkdir [file dirname $area_rpt]
file mkdir [file dirname $power_rpt]

#======================================================================
# 2. Initialization & Search Paths
#======================================================================
# Set library search paths
# Here, you can add the path that contains your library files
# set_app_var search_path [concat $::env(search_path) "/path/to/your/lib"]

# Set target and link libraries
set_app_var target_library $std_cell_lib
set_app_var link_library   "* $std_cell_lib"

#======================================================================
# 3. Analyze and Elaborate Design (RTL) and Libraries
#======================================================================
# Analyze Verilog source files
foreach file $verilog_list {
    analyze -format verilog $file
}

# Elaborate the top-level module
elaborate $top_name

# MARK: New added
ungroup -start_level 2 -all


#======================================================================
# 4. Set Top Design & Constraints
#======================================================================
# Optional: Read SDC file if you have a separate constraint file
# read_file -format sdc "/path/to/constraints.sdc"
# If there is no separate SDC file, you can create clocks and IO constraints here
current_design $top_name


create_clock -name clock -period $clk_period [get_ports clk]


# set the async reset to false path
set_false_path -from [get_ports resetn]


# set the delay percentage to the clock period (20%)
set delay_percent 0
set max_input_delay  [expr {$clk_period * $delay_percent}]
set max_output_delay [expr {$clk_period * $delay_percent}]

# set the max input delay
set_input_delay -clock clock -max $max_input_delay [all_inputs]

# set the min input delay
set_input_delay -clock clock -min 0.0 [all_inputs]

# set the max output delay
set_output_delay -clock clock -max $max_output_delay [all_outputs]

# set the min output delay
set_output_delay -clock clock -min 0.0 [all_outputs]

# report the clocks
report_clocks


#======================================================================
# 5. Synthesis & Optimization
#======================================================================
# Check design integrity
check_design


# Run compile. "compile_ultra" is recommended for better QoR
# compile
# compile_ultra

# MARK: New added
compile_ultra -no_autoungroup

#======================================================================
# 6. Generate and Export Synthesis Results
#======================================================================
# # Write out the synthesized netlist (Verilog format)
# write_file -format verilog -output $netlist_out

# # Write out the final timing constraints (SDC format)
# write_sdc $sdc_out

# # Write out the DDC database for further steps (e.g., P&R)
# write_file -format ddc -output $mapped_ddc_out

# # Generate synthesis reports

# report_timing > $timing_rpt
# report_area   > $area_rpt
# report_power  > $power_rpt



write -format verilog -hierarchy -output $netlist_out
write -format ddc     -hierarchy -output $mapped_ddc_out
write_sdc -nosplit $sdc_out

report_timing -nosplit -transition_time -nets -attributes > $timing_rpt
report_area   -nosplit -hierarchy > $area_rpt
report_power  -nosplit -hierarchy > $power_rpt



#======================================================================
# 7. Exit
#======================================================================
exit
