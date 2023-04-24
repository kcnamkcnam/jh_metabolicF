import statistics
import os
from datetime import datetime
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
import ast
import re
from sklearn.neighbors import KNeighborsRegressor

def get_mode(number_list):
    try:
        return "The mode of the numbers is {}".format(statistics.mode(number_list))
    except statistics.StatisticsError as exc:
        return "Error calculating mode: {}".format(exc)

def add_dropdown_menu(process_dir):
    # this function will read 'model.txt' file and add more models to the dropdown menu list
    # and then it will create a new index file (index_new.html)

    model_file = process_dir + "model.txt"
    index_file = process_dir + "index.html"
    new_index_file = process_dir + "index_new.html"
    search_txt = "class=\"dropdown-item\">"  # dropdown menu for search
    pre_txt = "<li><a class=\"dropdown-item\">"
    post_txt = "</a></li>\n"
    
    if os.path.exists(new_index_file):  # remove existing "index_new.html" file
      os.remove(new_index_file)

    if os.path.exists(model_file): # if model file exists, add more models for the dropdown menu.
      with open(index_file, 'r') as infile, open(model_file, 'r') as infile1, open(new_index_file, 'w') as outfile:
        saved_line = ""
        for line in infile:
            if re.search(search_txt, line): # search for the dropdown menu
                if saved_line: # both current and previous lines are dropdown menu, write the saved one.
                    outfile.write(saved_line)
                saved_line = line  # save the current dropdown menu and get the next one.
                continue
            else:
                if saved_line:
                    outfile.write(saved_line)
                    for model in infile1: # read model.txt file and add more models to the dropdown menu.
                        model_add = model.strip() # remove all white spaces.
                        if (model_add):
                            indentation = ""
                            for char in saved_line: # find out indentation for the dropdown menu item.
                                if char != '<':
                                    indentation += char
                                else:
                                    break
                            outfile.write(indentation + pre_txt + model_add + post_txt)
                    saved_line = ""
                outfile.write(line)

def d2b(d, n):
    d = np.array(d)
    d = np.reshape(d, (1, -1))
    power = np.flipud(2**np.arange(n))

    g = np.zeros((np.shape(d)[1], n))

    for i, num in enumerate(d[0]):
        g[i] = num * np.ones((1,n))
    b = np.floor((g%(2*power))/power)
    b = np.fliplr(b)
    return b

def process_input(input_file, sheets_str):
    home_dir = os.getcwd()
    process_dir = os.getcwd() + "/app/jupyter/"

    os.chdir(process_dir)
    input_file.save(process_dir + input_file.filename) #upload user input file.
    xlsname = input_file.filename #user input file name

    xmlname='simple1.xml' #specify this for each model
    lmid=4 #specify this for each model
    tracer = "input" #fixed variable for the excel sheet where tracers are

    #when the user enters "None" or Nothing in sheets field, set the default value "None"
    #sheets_tmp = sheets_str.replace('"', '') # remove all quotation marks
    #if sheets_str == "" or sheets_tmp.upper() == "NONE" :
    #    sheets = None
    #    df2 = pd.read_excel(xlsname, sheet_name=sheets, header=None)
    #else:
        #sheets = ["1,3_13C", "2", "3"]
        #sheets = re.findall(r'"(.*?)"', sheets_str) #convert string input to list type.
    sheets = ast.literal_eval('[' + sheets_str + ']') #convert string input to list type.
    df2 = pd.read_excel(xlsname, sheet_name=sheets, header=None)
    inp = pd.read_excel(xlsname, sheet_name=tracer, header=None)

    #exp = ["U", "C1", "C2"] #user input
    #exp = re.findall(r'"(.*?)"', exp_str) #convert string input to list type.
    #exp = ast.literal_eval('[' + exp_str + ']') #convert string input to list type.

    #code for placing the data into the correct spot
    tree = et.parse(xmlname)
    root=tree.getroot()

    a=range(1,2**inp.index.stop-1)
    alltracers=d2b(a,inp.index.stop)

    mat=np.ones((len(root[1][0]),lmid*(2**inp.index.stop-2)))*-1
    inpmetind=[]

    for a in sheets:
        lab=df2[a].loc[0:,1:].mean(1)
        metind=np.where(df2[a][0].notnull())[0]
        l=len(metind)
        for i, b in enumerate(metind):
            if i<l-1:
                mea=lab[metind[i]:metind[i+1]]
            else:
                mea=lab[metind[i]:]
            for j in inp.index:
                inpmet=inp[0][j]
                labeling=inpmet[inpmet.index('__'):]
                expind=np.where((alltracers==inp.loc[j,1:].to_numpy()).all(axis=1))
                inpmetname=inpmet[:inpmet.index('__')]
                for k, c in enumerate(root[1][0]):
                    if labeling in df2[a][0][b] and c.attrib['id'] in df2[a].loc[b,0]:
                        mat[k,lmid*expind[0].item():lmid*expind[0].item()+lmid]=np.zeros((1,lmid))
                        mat[k,lmid*expind[0].item():lmid*expind[0].item()+len(mea)]=mea
                    if c.attrib['id'] in inpmet[:inpmet.index('__')] and k not in inpmetind:
                        inpmetind.append(k)

    vec=np.delete(mat,inpmetind,0).reshape(-1)
    
    # clean-up; remove the uploaded input file
    if os.path.exists(xlsname):
        os.remove(xlsname)
    os.chdir(home_dir)
    
    return vec
    
def process_data(input_data):
    result = ""
    for line in input_data.splitlines():
        if line != "":
            numbers = [float(n.strip()) for n in line.split(",")]
            result += str(sum(numbers))
        result += "\n"
    return result

def do_addition(number1, number2):
    return number1 + number2

def imputeLabelsFromScratched(scratchedLabelSet,modelType):
    home_dir = os.getcwd()
    training_dir = os.getcwd() + "/app/ML-Trainings/"
    # Prepare training labels based on which model is used
    if modelType == 'Simple':
        label_train = np.loadtxt(training_dir + "labeling_Simple_seed0_20230206.txt")
    elif modelType == 'UpperGly':
        label_train = np.loadtxt(training_dir + "labeling_UpperGly_13C_seed0_20230112.txt")
    elif modelType == 'Gly13C2H':
        label_train = np.loadtxt(training_dir + "labeling_Gly_13C2H_seed0_20230112.txt")
    elif modelType == 'GlyPPP':
        label_train = np.loadtxt(training_dir + "labeling_GlyPPP_seed0_20230118.txt")
    elif modelType == 'MammalianCCM':
        label_train = np.loadtxt(training_dir + "labeling_1of4_MammalianCCM_seed0.txt")
        label_train = np.concatenate((label_train,np.loadtxt(training_dir + 'labeling_2of4_MammalianCCM_seed0.txt')))
        label_train = np.concatenate((label_train,np.loadtxt(training_dir + 'labeling_3of4_MammalianCCM_seed0.txt')))
        label_train = np.concatenate((label_train,np.loadtxt(training_dir + 'labeling_4of4_MammalianCCM_seed0.txt')))
    else: raise ValueError("Unexpected model name. Accepted models are 'Simple','UpperGly','Gly13C2H','GlyPPP', or 'MammalianCCM'")
    
    scratchedLabelSet=np.array(scratchedLabelSet)

    n_label = len(scratchedLabelSet)

    #missing_list = np.argwhere(np.isnan(scratchedLabelSet))
    missing_list = np.argwhere(scratchedLabelSet<0)
    missing_list = missing_list.reshape(len(missing_list))
    known_list = list(range(n_label))
    [known_list.remove(x) for x in missing_list]

    X = label_train[:,known_list]
    y = label_train[:,missing_list]

    neigh = KNeighborsRegressor(n_neighbors=2, weights='distance')
    neigh.fit(X, y)

    pred = neigh.predict(scratchedLabelSet[known_list].reshape(1, -1))
    neigh_dist, neigh_idx = neigh.kneighbors(scratchedLabelSet[known_list].reshape(1, -1))

    #print("neighbor distance: ", neigh_dist)
    #print("neighbor index: ", neigh_idx)

    label_all = np.zeros(n_label)

    label_all[known_list] = scratchedLabelSet[known_list] 
    label_all[missing_list] = pred
    fullLabelSet = label_all.reshape(1,n_label)

    #np.savetxt("label_pred.txt", fullLabelSet, fmt="%.6f")
    
    return fullLabelSet

def predictFluxesFromLabels(fullLabelSet,modelType):
    from keras.models import model_from_json
    home_dir = os.getcwd()
    model_dir = os.getcwd() + "/app/ML-Models/"
    # Prepare function calls based on which model is used
    if modelType == 'Simple':
        modelArchitectureFile = model_dir+'Simple_ANN_C2-4.json'
        modelWeightsFiles = model_dir+'Simple_ANN_C2-4.h5'
        net_flux = [0,1]
        exchange_flux = [2,3,4,5]
        freeList = ['Ex_E','Ex_F','v2','v3','v4','v5']
        fullList = ['v1','v2','v3','v4','v5','Ex_E','Ex_F','v1_x','v2_x','v3_x','v4_x','v_5_x','Ex_E_x','Ex_F_x']
        kernelNet = np.loadtxt(model_dir+"KernelNet_Simple.txt")
        kernelXch = np.loadtxt(model_dir+"KernelXch_Simple.txt")
    elif modelType == 'UpperGly':
        modelArchitectureFile = model_dir+'UpperGly_13C_seed0_20230112_ANN_C2-4.json'
        modelWeightsFiles = model_dir+'UpperGly_13C_seed0_20230112_ANN_C2-4.h5'
        net_flux = [0,1,2]
        exchange_flux = [3,4,5,6]
        freeList = ['G6P_EX','F6P_EX','DHAP_EX','PGI','PFK','FBA','TPI']
        fullList = ['G6P_EX','F6P_EX','DHAP_EX','PGI','PFK','FBA','TPI']
        kernelNet = np.diag(np.ones(3))
        kernelXch = np.diag(np.ones(4))
    elif modelType == 'Gly13C2H':
        modelArchitectureFile = model_dir+'Gly_13C2H_seed0_ANN_C2-4.json'
        modelWeightsFiles = model_dir+'Gly_13C2H_seed0_ANN_C2-4.h5'
        net_flux = [0,1,2,3,4,5,6,7]
        exchange_flux = [8,9,10,11,12,13,14,15,16,17]
        freeList = ['H_EX','NADH_EX','G6P_EX','F6P_EX','DHAP_EX','PG3_EX','PEP_EX','PYR_EX','PGI','PFK','FBA','TPI','GAPD','PGK','PGM1','PGM2','ENO','PYK']
        fullList = ['GLC_IN','H_IN','NADH_IN','PGI','PFK','FBA','TPI','GAPD','PGK','PGM1','PGM2','ENO','PYK','H_EX','NADH_EX','G6P_EX','F6P_EX','DHAP_EX','PG3_EX','PEP_EX',
                    'GLC_IN','H_IN','NADH_IN','PGI','PFK','FBA','TPI','GAPD','PGK','PGM1','PGM2','ENO','PYK','H_EX','NADH_EX','G6P_EX','F6P_EX','DHAP_EX','PG3_EX','PEP_EX']
        kernelNet = np.loadtxt(model_dir+"KernelNet_Gly13C2H.txt")
        kernelXch = np.loadtxt(model_dir+"KernelXch_Gly13C2H.txt")
    elif modelType == 'GlyPPP':
        modelArchitectureFile = model_dir+'GlyPPP_seed0_ANN_C2-4.json'
        modelWeightsFiles = model_dir+'GlyPPP_seed0_ANN_C2-4.h5'
        net_flux = [0,1,2,3,4,5,6,7,8,9,10,11,12]
        exchange_flux = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        freeList = ['TAL','SBPASE','CO2_EX','H_EX','NADPH_EX','NADH_EX','G6P_EX','F6P_EX','DHAP_EX','R5P_EX','E4P_EX','PG3_EX','PEP_EX'
                    'PGI','PFK','FBA','TPI','GAPD','PGK','PGM1','PGM2','ENO','G6PDH','GND','RPI','RPE','TKT2','TKT1','TAL','SBA','SBPASE','PGI_leak','RPI_leak']
        fullList = ['GLC_IN','CO2_IN','H_IN','NADPH_IN','NADH_IN','PGI','PFK','FBA','TPI','GAPD','PGK','PGM1','PGM2','ENO','G6PDH','GND','RPI','RPE','TKT2','TKT1','TAL','SBA','SBPASE','CO2_EX','H_EX','NADPH_EX','NADH_EX','G6P_EX','F6P_EX','DHAP_EX','R5P_EX',
                    'E4P_EX','PG3_EX','PEP_EX','PGI_leak','RPI_leak',
                    'GLC_IN','CO2_IN','H_IN','NADPH_IN','NADH_IN','PGI','PFK','FBA','TPI','GAPD','PGK','PGM1','PGM2','ENO','G6PDH','GND','RPI','RPE','TKT2','TKT1','TAL','SBA','SBPASE','CO2_EX','H_EX','NADPH_EX','NADH_EX','G6P_EX','F6P_EX','DHAP_EX','R5P_EX','E4P_EX','PG3_EX','PEP_EX','PGI_leak','RPI_leak']
        kernelNet = np.loadtxt(model_dir+"KernelNet_GlyPPP.txt")
        kernelXch = np.loadtxt(model_dir+"KernelXch_GlyPPP.txt")
    elif modelType == 'MammalianCCM':
        modelArchitectureFile = model_dir+'MammalianCCM_ANN_C2-4.json'
        modelWeightsFiles = model_dir+'MammalianCCM_seed0_ANN_C2-4.h5'
        net_flux = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        exchange_flux = [21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
        freeList = ['EX_CO2','EX_DHAP','EX_PGA','EX_PYR','EX_LAC','EX_G6P','pc','tal','SBPase','EX_R5P','EX_OAA','IN_AC','fum','mdh','OGA_Glu','IN_Glu','OGA_Gln','EX_Gln','IN_OAA','EX_OGA_Glu','EX_AC_cyt',
                    'hk','pgi','pfk','fba','tpi','gapd','pgk','eno','pyk','ldh','ppck','me','pc','g6pdh','gnd','rpi','rpe','tkt2','tkt1','tal','SBA','SBPase','pdh','cs','acitl','icdh','akgdh','sucoas','sucd','fum','mdh','PYR_Ala','OGA_Glu','OGA_Gln']
        fullList = ['IN_GLC','IN_CO2','EX_CO2','hk','pgi','pfk','fba','tpi','gapd','pgk','eno','pyk','ldh','EX_DHAP','EX_PGA','EX_PYR','EX_LAC','EX_G6P','ppck','me','pc','g6pdh','gnd','rpi','rpe','tkt2','tkt1','tal','SBA','SBPase','EX_R5P','EX_OAA','pdh','IN_AC','cs','acitl','icdh','akgdh','sucoas','sucd','fum','mdh','PYR_Ala','EX_PYR_Ala','OGA_Glu','IN_Gln','IN_Glu','OGA_Gln','EX_Gln','IN_OAA','EX_OGA_Glu','EX_AC_cyt',
                    'IN_GLC','IN_CO2','EX_CO2','hk','pgi','pfk','fba','tpi','gapd','pgk','eno','pyk','ldh','EX_DHAP','EX_PGA','EX_PYR','EX_LAC','EX_G6P','ppck','me','pc','g6pdh','gnd','rpi','rpe','tkt2','tkt1','tal','SBA','SBPase','EX_R5P','EX_OAA','pdh','IN_AC','cs','acitl','icdh','akgdh','sucoas','sucd','fum','mdh','PYR_Ala','EX_PYR_Ala','OGA_Glu','IN_Gln','IN_Glu','OGA_Gln','EX_Gln','IN_OAA','EX_OGA_Glu','EX_AC_cyt']
        kernelNet = np.loadtxt(model_dir+"KernelNet_MammalianCCM.txt")
        kernelXch = np.loadtxt(model_dir+"KernelXch_MammalianCCM.txt")
    else: raise ValueError("Unexpected model name. Accepted models are 'Simple','UpperGly','Gly13C2H','GlyPPP', or 'MammalianCCM'")
    
    # Load model architecture
    json_file = open(modelArchitectureFile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loadedModel = model_from_json(loaded_model_json)

    # Load weights into model
    loadedModel.load_weights(modelWeightsFiles) 

    # Compile model and optimizer
    loadedModel.compile(optimizer='adam', loss='mae', metrics=[None])
    
    # Predict fluxes and transform output into real flux values
    freeFluxes = loadedModel.predict(fullLabelSet)
    freeFluxes[:,net_flux] = np.piecewise(freeFluxes[:,net_flux],[freeFluxes[:,net_flux]<0,freeFluxes[:,net_flux]<0.97997,freeFluxes[:,net_flux]>=0.97997],[-4.5,lambda freeFluxes: np.log(freeFluxes/(1-freeFluxes)),lambda freeFluxes: 4**freeFluxes])
    freeFluxes[:,exchange_flux] = np.piecewise(freeFluxes[:,exchange_flux],[freeFluxes[:,exchange_flux]>-4,(-5<freeFluxes[:,exchange_flux])&(freeFluxes[:,exchange_flux]<=-4),freeFluxes[:,exchange_flux]<=-5],[lambda freeFluxes: 10**freeFluxes,lambda freeFluxes: (freeFluxes+5)/10000,0])
    freeFluxes[:,exchange_flux] = np.maximum(0, freeFluxes[:,exchange_flux]) # Ensures all exchange fluxes are positive

    # Generate the full set of net and exchange fluxes
    fullNet = np.matmul(kernelNet,np.transpose(freeFluxes[:,net_flux]))
    fullXch = np.matmul(kernelXch,np.transpose(freeFluxes[:,exchange_flux]))
    fullFluxes = np.concatenate((fullNet,fullXch))

    return freeFluxes, freeList, fullFluxes, fullList