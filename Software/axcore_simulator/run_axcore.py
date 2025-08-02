from re import I
import pandas
import configparser
import os
import numpy as np
import AxCore.src.benchmarks.benchmarks as benchmarks
from AxCore.src.simulator.stats import Stats
from AxCore.src.simulator.simulator import AxCoreSimulator
from bitfusion.src.simulator.simulator import Simulator
from AxCore.src.sweep.sweep import check_pandas_or_run_ax
from bitfusion.src.sweep.sweep import check_pandas_or_run
from bitfusion.src.utils.utils import *
from bitfusion.src.optimizer.optimizer import optimize_for_order, get_stats_fast
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--synth_csv', type=str, default='./results/systolic_array_synth.csv', help='Path to systolic array synthesis results csv')
args = parser.parse_args()

synth_filename = os.path.basename(args.synth_csv)
config_name = os.path.splitext(synth_filename)[0]

def df_to_stats(df):
    stats = Stats()
    stats.total_cycles = float(df['Cycles'].iloc[0])
    stats.mem_stall_cycles = float(df['Memory wait cycles'].iloc[0])
    stats.reads['act'] = float(df['IBUF Read'].iloc[0])
    stats.reads['out'] = float(df['OBUF Read'].iloc[0])
    stats.reads['wgt'] = float(df['WBUF Read'].iloc[0])
    stats.reads['dram'] = float(df['DRAM Read'].iloc[0])
    stats.writes['act'] = float(df['IBUF Write'].iloc[0])
    stats.writes['out'] = float(df['OBUF Write'].iloc[0])
    stats.writes['wgt'] = float(df['WBUF Write'].iloc[0])
    stats.writes['dram'] = float(df['DRAM Write'].iloc[0])
    return stats

sim_sweep_columns = ['N', 'M',
        'Max Precision (bits)', 'Min Precision (bits)',
        'Network', 'Layer',
        'Cycles', 'Memory wait cycles',
        'WBUF Read', 'WBUF Write',
        'OBUF Read', 'OBUF Write',
        'IBUF Read', 'IBUF Write',
        'DRAM Read', 'DRAM Write',
        'Bandwidth (bits/cycle)',
        'WBUF Size (bits)', 'OBUF Size (bits)', 'IBUF Size (bits)',
        'Batch size']

batch_size = 32

results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
# AxCore configuration file
config_file = 'conf_axcore.ini'
# Create simulator object
bf_e_sim = AxCoreSimulator(config_file, synth_csv=args.synth_csv, verbose=False)
# bf_e_sim = Simulator(config_file, False)
bf_e_sim_sweep_csv = os.path.join(results_dir, 'axcore.csv')
bf_e_sim_sweep_df = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results = check_pandas_or_run_ax(bf_e_sim, bf_e_sim_sweep_df, bf_e_sim_sweep_csv, batch_size=batch_size, bench_type='axcore', weight_stationary=True)
bf_e_results = bf_e_results.groupby('Network',as_index=False).agg(np.sum)
bf_e_cycles_axcore = []
bf_e_energy_axcore = []
for name in benchmarks.benchlist:
    bf_e_stats = df_to_stats(bf_e_results.loc[bf_e_results['Network'] == name])
    bf_e_cycles_axcore.append(bf_e_stats.total_cycles)
    bf_e_energy_axcore.append(bf_e_stats.get_energy_breakdown(bf_e_sim.get_energy_cost()))


# FIGNA weight_stationary configuration file
config_file = 'conf_figna.ini'
# Create simulator object
bf_e_sim1 = AxCoreSimulator(config_file, synth_csv=args.synth_csv, verbose=False)
bf_e_sim_sweep_csv_1 = os.path.join(results_dir, 'figna.csv')
bf_e_sim_sweep_df_1 = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results_1 = check_pandas_or_run_ax(bf_e_sim1, bf_e_sim_sweep_df_1, bf_e_sim_sweep_csv_1, batch_size=batch_size, bench_type='axcore', weight_stationary=True)
bf_e_results_1 = bf_e_results_1.groupby('Network',as_index=False).agg(np.sum)
# area_stats = bf_e_sim.get_area()
bf_e_cycles_figna = []
bf_e_energy_figna = []
for name in benchmarks.benchlist:
    bf_e_stats_1 = df_to_stats(bf_e_results_1.loc[bf_e_results_1['Network'] == name])
    bf_e_cycles_figna.append(bf_e_stats_1.total_cycles)
    bf_e_energy_figna.append(bf_e_stats_1.get_energy_breakdown(bf_e_sim1.get_energy_cost()))
 
# FPE configuration file
config_file = 'conf_fpe.ini'
# Create simulator object   
bf_e_sim2 = AxCoreSimulator(config_file, synth_csv=args.synth_csv, verbose=False)
bf_e_sim_sweep_csv_2 = os.path.join(results_dir, 'fpe.csv')
bf_e_sim_sweep_df_2 = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results_2 = check_pandas_or_run_ax(bf_e_sim2, bf_e_sim_sweep_df_2, bf_e_sim_sweep_csv_2, batch_size=batch_size, bench_type='axcore', weight_stationary=True)
bf_e_results_2 = bf_e_results_2.groupby('Network',as_index=False).agg(np.sum)
# area_stats = bf_e_sim.get_area()
bf_e_cycles_fpe = []
bf_e_energy_fpe = []
for name in benchmarks.benchlist:
    bf_e_stats_2 = df_to_stats(bf_e_results_2.loc[bf_e_results_2['Network'] == name])
    bf_e_cycles_fpe.append(bf_e_stats_2.total_cycles)
    bf_e_energy_fpe.append(bf_e_stats_2.get_energy_breakdown(bf_e_sim2.get_energy_cost()))
    
# FPMA weight_stationary configuration file
config_file = 'conf_fpma.ini'
# Create simulator object
bf_e_sim3 = AxCoreSimulator(config_file, synth_csv=args.synth_csv, verbose=False) 
bf_e_sim_sweep_csv_3 = os.path.join(results_dir, 'fpma.csv')
bf_e_sim_sweep_df_3 = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results_3 = check_pandas_or_run_ax(bf_e_sim3, bf_e_sim_sweep_df_3, bf_e_sim_sweep_csv_3, batch_size=batch_size, bench_type='axcore', weight_stationary=True)
bf_e_results_3 = bf_e_results_3.groupby('Network',as_index=False).agg(np.sum)
# area_stats = bf_e_sim.get_area()
bf_e_cycles_fpma = []
bf_e_energy_fpma = []
for name in benchmarks.benchlist:
    bf_e_stats_3 = df_to_stats(bf_e_results_3.loc[bf_e_results_3['Network'] == name])
    bf_e_cycles_fpma.append(bf_e_stats_3.total_cycles)
    bf_e_energy_fpma.append(bf_e_stats_3.get_energy_breakdown(bf_e_sim3.get_energy_cost()))

# FGLUT weight_stationary configuration file
config_file = 'conf_figlut.ini'
# Create simulator object
bf_e_sim4 = AxCoreSimulator(config_file, synth_csv=args.synth_csv, verbose=False)
bf_e_sim_sweep_csv_4 = os.path.join(results_dir, 'figlut.csv')
bf_e_sim_sweep_df_4 = pandas.DataFrame(columns=sim_sweep_columns)
bf_e_results_4 = check_pandas_or_run_ax(bf_e_sim4, bf_e_sim_sweep_df_4, bf_e_sim_sweep_csv_4, batch_size=batch_size, bench_type='axcore', weight_stationary=True)
bf_e_results_4 = bf_e_results_4.groupby('Network',as_index=False).agg(np.sum)
# area_stats = bf_e_sim.get_area()
bf_e_cycles_figlut = []
bf_e_energy_figlut = []
for name in benchmarks.benchlist:
    bf_e_stats_4 = df_to_stats(bf_e_results_4.loc[bf_e_results_4['Network'] == name])
    bf_e_cycles_figlut.append(bf_e_stats_4.total_cycles)
    bf_e_energy_figlut.append(bf_e_stats_4.get_energy_breakdown(bf_e_sim4.get_energy_cost()))

# print(bf_e_cycles_axcore)
# print(bf_e_cycles_figna)
# print(bf_e_cycles_fpma)
# print(bf_e_cycles_fpe)

all_cyc = []
cyc_1_mean = 0
cyc_2_mean = 0
cyc_3_mean = 0
cyc_4_mean = 0
cyc_5_mean = 0
cyc_6_mean = 0
    
# write to csv
model_name_dict = {'vgg16':'VGG16', 
                   'resnet18':'ResNet18',
                   'resnet50':'ResNet50',
                   'inceptionv3':'InceptionV3',
                   'vit':'ViT',
                   'mnli':'BERT-MNLI',
                   'cola':'BERT-CoLA',
                   'sst_2':'BERT-SST-2',
                   'opt_125m':'Opt125M',
                   'opt_350m':'Opt350M',
                   'opt_1_3b':'Opt1.3B',
                   'opt_2_7b':'Opt2.7B',
                   'opt_6_7b':'Opt6.7B',
                   'opt_13b':'Opt13B',
                   'opt_30b':'Opt30B',
                   'opt_66b':'Opt66B'}

ff = open(os.getcwd() + '/results/axcore_res.csv', "a")
ff.write(f"Configuration: {config_name}\n")
wr_line = "Time, "
wr_bench_name = ", "
wr_model_name = ", "
for i in range(len(bf_e_cycles_axcore)):
    model_name = benchmarks.benchlist[i]
    
    cyc_3 = bf_e_cycles_fpe[i]
    cyc_1 = bf_e_cycles_axcore[i] / cyc_3
    cyc_1_mean += cyc_1
    cyc_2 = bf_e_cycles_figna[i] / cyc_3
    cyc_2_mean += cyc_2
    cyc_4 = bf_e_cycles_fpma[i] / cyc_3
    cyc_4_mean += cyc_4
    cyc_5 = bf_e_cycles_figlut[i] / cyc_3
    cyc_5_mean += cyc_5
    
    cyc_3 = cyc_3 / cyc_3
    cyc_3_mean += cyc_3
    
    all_cyc.append(cyc_1)
    all_cyc.append(cyc_2)
    all_cyc.append(cyc_3)
    all_cyc.append(cyc_4)
    all_cyc.append(cyc_5)
    
    if(model_name == 'vgg16' or model_name == 'resnet50'):
        wr_model_name += model_name_dict[model_name] + ", , , "
        wr_bench_name += "AxCore, FGLUT, FIGNA, FPMA, FPE, "
        wr_line += "%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1, cyc_5, cyc_2, cyc_4, cyc_3)
        print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1, cyc_5, cyc_2, cyc_4, cyc_3), end="")
    else:
        wr_model_name += model_name_dict[model_name] + ", , , "
        wr_bench_name += "AxCore, FGLUT, FIGNA, FPMA, FPE, "
        wr_line += "%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1, cyc_5, cyc_2, cyc_4, cyc_3)
        print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1, cyc_5, cyc_2, cyc_4, cyc_3), end="")
        
cyc_1_mean /= len(bf_e_cycles_axcore)
cyc_2_mean /= len(bf_e_cycles_axcore)
cyc_3_mean /= len(bf_e_cycles_axcore)
cyc_4_mean /= len(bf_e_cycles_axcore)
cyc_5_mean /= len(bf_e_cycles_axcore)

wr_model_name += "Geomean, , , , , \n"
wr_bench_name += "AxCore, FGLUT, FIGNA, FPMA, FPE, \n"
wr_line += ("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1_mean, cyc_5_mean, cyc_2_mean, cyc_4_mean, cyc_3_mean)) + "\n"
ff.write(wr_model_name)
ff.write(wr_bench_name)
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(cyc_1_mean, cyc_5_mean, cyc_2_mean, cyc_4_mean, cyc_3_mean))
print()


all_energy1 = []
all_energy2 = []
all_energy3 = []
all_energy4 = []
all_energy5 = []
for i in range(len(bf_e_cycles_axcore)):
    
    model_name = benchmarks.benchlist[i]
    
    energy_data_3 = bf_e_energy_fpe[i]
    
    energy_data_total = energy_data_3[0] + energy_data_3[1] + energy_data_3[2] + energy_data_3[3]
    
    energy_data_3[0] /= energy_data_total
    energy_data_3[1] /= energy_data_total
    energy_data_3[2] /= energy_data_total
    energy_data_3[3] /= energy_data_total
    
    energy_data_1 = bf_e_energy_axcore[i]
    energy_data_1[0] /= energy_data_total
    energy_data_1[1] /= energy_data_total
    energy_data_1[2] /= energy_data_total
    energy_data_1[3] /= energy_data_total
    
    energy_data_2 = bf_e_energy_figna[i]
    energy_data_2[0] /= energy_data_total
    energy_data_2[1] /= energy_data_total
    energy_data_2[2] /= energy_data_total
    energy_data_2[3] /= energy_data_total
    
    energy_data_4 = bf_e_energy_fpma[i]
    energy_data_4[0] /= energy_data_total
    energy_data_4[1] /= energy_data_total
    energy_data_4[2] /= energy_data_total
    energy_data_4[3] /= energy_data_total
    
    energy_data_5 = bf_e_energy_figlut[i]
    energy_data_5[0] /= energy_data_total
    energy_data_5[1] /= energy_data_total
    energy_data_5[2] /= energy_data_total
    energy_data_5[3] /= energy_data_total
    
    all_energy1.append(energy_data_1[0])
    all_energy1.append(energy_data_5[0])
    all_energy1.append(energy_data_2[0])
    all_energy1.append(energy_data_4[0])
    all_energy1.append(energy_data_3[0])

    all_energy2.append(energy_data_1[1])
    all_energy2.append(energy_data_5[1])
    all_energy2.append(energy_data_2[1])
    all_energy2.append(energy_data_4[1])
    all_energy2.append(energy_data_3[1])

    all_energy3.append(energy_data_1[2])
    all_energy3.append(energy_data_5[2])
    all_energy3.append(energy_data_2[2])
    all_energy3.append(energy_data_4[2])
    all_energy3.append(energy_data_3[2])

    all_energy4.append(energy_data_1[3])
    all_energy4.append(energy_data_5[3])
    all_energy4.append(energy_data_2[3])
    all_energy4.append(energy_data_4[3])
    all_energy4.append(energy_data_3[3])
    print(all_energy4)

print()

wr_line = "Static, "
for i in all_energy1:
    wr_line += "%0.2f, " %(i)
    print("%0.2f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0
energy_mean_5 = 0

biscale_offset = 0
for i in range(len(bf_e_cycles_axcore)):
    model_name = benchmarks.benchlist[i]
    idx = i * 5 + biscale_offset
    energy_mean_1 += all_energy1[idx]
    energy_mean_2 += all_energy1[idx+1]
    energy_mean_4 += all_energy1[idx+2]
    energy_mean_3 += all_energy1[idx+3]
    energy_mean_5 += all_energy1[idx+4]

energy_mean_1 /= len(bf_e_cycles_axcore)
energy_mean_2 /= len(bf_e_cycles_axcore)
energy_mean_4 /= len(bf_e_cycles_axcore)
energy_mean_3 /= len(bf_e_cycles_axcore)
energy_mean_5 /= len(bf_e_cycles_axcore)

wr_line += ("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(energy_mean_1, energy_mean_2, energy_mean_4, energy_mean_3, energy_mean_5)) + "\n"
ff.write("\n")
ff.write(wr_model_name)
ff.write(wr_bench_name)
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(energy_mean_1, energy_mean_2, energy_mean_4, energy_mean_3, energy_mean_5))

wr_line = "Dram, "
for i in all_energy2:
    wr_line += "%0.2f, " %(i)
    print("%0.2f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0
energy_mean_5 = 0

biscale_offset = 0
for i in range(len(bf_e_cycles_axcore)):
    model_name = benchmarks.benchlist[i]
    idx = i * 5 + biscale_offset
    energy_mean_1 += all_energy2[idx]
    energy_mean_2 += all_energy2[idx+1]
    energy_mean_4 += all_energy2[idx+2]
    energy_mean_3 += all_energy2[idx+3]
    energy_mean_5 += all_energy2[idx+4]

energy_mean_1 /= len(bf_e_cycles_axcore)
energy_mean_2 /= len(bf_e_cycles_axcore)
energy_mean_4 /= len(bf_e_cycles_axcore)
energy_mean_3 /= len(bf_e_cycles_axcore)
energy_mean_5 /= len(bf_e_cycles_axcore)

wr_line += ("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(energy_mean_1, energy_mean_2, energy_mean_4, energy_mean_3, energy_mean_5)) + "\n"
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(energy_mean_1, energy_mean_2, energy_mean_4, energy_mean_3, energy_mean_5))


wr_line = "Buffer, "
for i in all_energy3:
    wr_line += "%0.2f, " %(i)
    print("%0.2f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0
energy_mean_5 = 0

biscale_offset = 0
for i in range(len(bf_e_cycles_axcore)):
    model_name = benchmarks.benchlist[i]
    idx = i * 5 + biscale_offset
    energy_mean_1 += all_energy3[idx]
    energy_mean_2 += all_energy3[idx+1]
    energy_mean_4 += all_energy3[idx+2]
    energy_mean_3 += all_energy3[idx+3]
    energy_mean_5 += all_energy3[idx+4]

energy_mean_1 /= len(bf_e_cycles_axcore)
energy_mean_2 /= len(bf_e_cycles_axcore)
energy_mean_4 /= len(bf_e_cycles_axcore)
energy_mean_3 /= len(bf_e_cycles_axcore)

wr_line += ("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(energy_mean_1, energy_mean_2, energy_mean_4, energy_mean_3, energy_mean_5)) + "\n"
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(energy_mean_1, energy_mean_2, energy_mean_4, energy_mean_3, energy_mean_5))


wr_line = "Core, "
for i in all_energy4:
    wr_line += "%0.2f, " %(i)
    print("%0.2f, " %(i), end="")
energy_mean_1 = 0
energy_mean_2 = 0
energy_mean_3 = 0
energy_mean_4 = 0
energy_mean_5 = 0

biscale_offset = 0
for i in range(len(bf_e_cycles_axcore)):
    model_name = benchmarks.benchlist[i]
    idx = i * 5 + biscale_offset
    energy_mean_1 += all_energy4[idx]
    energy_mean_2 += all_energy4[idx+1]
    energy_mean_4 += all_energy4[idx+2]
    energy_mean_3 += all_energy4[idx+3]
    energy_mean_5 += all_energy4[idx+4]

energy_mean_1 /= len(bf_e_cycles_axcore)
energy_mean_2 /= len(bf_e_cycles_axcore)
energy_mean_4 /= len(bf_e_cycles_axcore)
energy_mean_3 /= len(bf_e_cycles_axcore)
energy_mean_5 /= len(bf_e_cycles_axcore)

wr_line += "%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(energy_mean_1, energy_mean_2, energy_mean_4, energy_mean_3, energy_mean_5) + "\n"
ff.write(wr_line)
wr_line = ""
print("%0.2f, %0.2f, %0.2f, %0.2f, %0.2f, " %(energy_mean_1, energy_mean_2, energy_mean_4, energy_mean_3, energy_mean_5))
ff.write(f"\n")
print("Please see the results at ./results/axcore_res.csv ")
ff.close()
