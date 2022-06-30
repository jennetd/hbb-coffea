#!/usr/bin/python
import os, sys
import subprocess

#Define the list of threshold
ddb1_thresholds = [0., 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
       0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
       0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
       0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
       0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
       0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
       0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
       0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
       0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
       0.99, 1. ]

ddc2_thresholds = [0.        , 0.001     , 0.00107227, 0.00114976, 0.00123285,
       0.00132194, 0.00141747, 0.00151991, 0.00162975, 0.00174753,
       0.00187382, 0.00200923, 0.00215443, 0.00231013, 0.00247708,
       0.00265609, 0.00284804, 0.00305386, 0.00327455, 0.00351119,
       0.00376494, 0.00403702, 0.00432876, 0.00464159, 0.00497702,
       0.0053367 , 0.00572237, 0.00613591, 0.00657933, 0.0070548 ,
       0.00756463, 0.00811131, 0.00869749, 0.00932603, 0.01      ,
       0.01072267, 0.01149757, 0.01232847, 0.01321941, 0.01417474,
       0.01519911, 0.01629751, 0.01747528, 0.01873817, 0.02009233,
       0.02154435, 0.0231013 , 0.02477076, 0.02656088, 0.02848036,
       0.03053856, 0.03274549, 0.03511192, 0.03764936, 0.04037017,
       0.04328761, 0.04641589, 0.04977024, 0.05336699, 0.05722368,
       0.06135907, 0.06579332, 0.07054802, 0.07564633, 0.08111308,
       0.0869749 , 0.09326033, 0.1       , 0.10722672, 0.1149757 ,
       0.12328467, 0.13219411, 0.14174742, 0.15199111, 0.16297508,
       0.17475284, 0.18738174, 0.2009233 , 0.21544347, 0.23101297,
       0.24770764, 0.26560878, 0.28480359, 0.30538555, 0.32745492,
       0.35111917, 0.37649358, 0.40370173, 0.43287613, 0.46415888,
       0.49770236, 0.53366992, 0.57223677, 0.61359073, 0.65793322,
       0.70548023, 0.75646333, 0.81113083, 0.869749  , 0.93260335,
       1.        ]

class Found(Exception): pass #for breaking nested loops


os.system("mkdir -p logs")
os.system("cp tar_ball/tar_ball.tar.gz logs/")
os.system("cp tar_ball/CMSSW_10_2_13.tar logs/")
current_path =  os.environ['PWD']
log_dir = "logs"

n_chunk = 10 #Process 10 points at a time
chunk_index = 0 #Chunk counter to keep index of the chunks
n_test = 2 #number of test loop

counter = 0
pair_string = ' '
#Loop over all the threshold combinations

try:
       for ddb1 in ddb1_thresholds:
              for ddc2 in ddc2_thresholds:

                     #Create the pairings
                     pair_string += '"{} {}" '.format(ddb1, ddc2)
                     counter += 1

                     if counter % n_chunk == 0:
                            
                            #Write the pre_scan
                            pre_scan_temp = open("{}/pre_scan_temp.sh".format(current_path)) #Template
                            pre_scan_local = "{}/{}/pre_scan_{}.sh".format(current_path, log_dir,chunk_index)

                            print("Creating pre scan files: ...")
                            pre_scan_file = open(pre_scan_local,"w")
                            for line in pre_scan_temp:
                                   line=line.replace('THRES_LIST', pair_string)
                                   pre_scan_file.write(line)
                            pre_scan_file.close()

                            #Write the main_scan
                            main_scan_temp = open("{}/main_scan_temp.sh".format(current_path)) #Template
                            main_scan_local = "{}/{}/main_scan_{}.sh".format(current_path, log_dir, chunk_index)

                            print("Creating main scan files: ...")
                            main_scan_file = open(main_scan_local, "w")
                            for line in main_scan_temp:
                                   line=line.replace('THRES_LIST', pair_string)
                                   main_scan_file.write(line)
                            main_scan_file.close()

                            #Produce excecutable:
                            print("Producing executable ...")
                            exec_temp =  open("{}/condor_temp.sh".format(current_path))
                            exec_local =  "{}/{}/condor_local_{}.sh".format(current_path, log_dir, chunk_index)

                            exec_local_file = open(exec_local, 'w')
                            for line in exec_temp:
                                   line=line.replace('PRE_SCAN_FILE', "pre_scan_{}.sh".format(chunk_index))
                                   line=line.replace('MAIN_SCAN_FILE', "main_scan_{}.sh".format(chunk_index))
                                   exec_local_file.write(line)
                            exec_local_file.close()


                            #Produce condor file
                            condor_temp = open("{}/condor_temp.sub".format(current_path))
                            condor_local = "{}/{}/condor_local_{}.sub".format(current_path, log_dir, chunk_index)

                            print("Creating condor files")
                            condor_local_file = open(condor_local, "w")
                            for line in condor_temp:
                                   line=line.replace('PREFIX', "condor_local_{}".format(chunk_index))
                                   line=line.replace('PRE_SCAN', "pre_scan_{}.sh".format(chunk_index))
                                   line=line.replace('MAIN_SCAN', "main_scan_{}.sh".format(chunk_index))
                                   line=line.replace('OUT_FILE', "out_file_{}.out".format(chunk_index))
                                   condor_local_file.write(line)
                            condor_local_file.close()

                            print("COUNTER: ", counter)
                            print("PAIRS: ", pair_string)
                            pair_string = ' '

                            chunk_index += 1

                            condor_temp.close()
                            exec_temp.close()
                            main_scan_temp.close()
                            pre_scan_temp.close()

                            os.system('condor_submit {}'.format(condor_local))

                     if chunk_index == n_test:
                            raise Found

except Found:
       print("BREAKING")