import statistics
import os
from datetime import datetime
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as et
import ast


def get_mode(number_list):
    try:
        return "The mode of the numbers is {}".format(statistics.mode(number_list))
    except statistics.StatisticsError as exc:
        return "Error calculating mode: {}".format(exc)

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

def process_input(input_file, sheets_str, exp_str):
    home_dir = os.getcwd()
    process_dir = os.getcwd() + "/app/jupyter/"
    output_filename = f'result_{str(datetime.now().strftime("%Y%m%d-%H%M%S"))}.csv'
    
    os.chdir(process_dir)
    input_file.save(process_dir + input_file.filename) #upload user input file.
    xlsname = input_file.filename #user input file name

    #sheets = ["1,3_13C", "2", "3"]
    #when the user enters "None" or Nothing for the sheets, set the default value "None"
    sheets_tmp = sheets_str.replace('"', '') # remove all ""
    if sheets_tmp == "" or sheets_tmp.upper() == "NONE" :
        sheets = None
    else:
        sheets_tuple = ast.literal_eval(sheets_str) #convert string input to list type.
        sheets = list(sheets_tuple) # convert to list
    
    #exp = ["U", "C1", "C2"] #user input
    exp_tuple = ast.literal_eval(exp_str) #convert string input to list type.
    exp = list(exp_tuple) # convert to list

    xmlname='simple1.xml' #specify this for each model
    lmid=4 #specify this for each model
    tracer = "input" #fixed variable for the excel sheet where tracers are

    #code for reading and placing the data
    df2 = pd.read_excel(xlsname, sheet_name=sheets, header=None)
    inp = pd.read_excel(xlsname, sheet_name=tracer, header=None)
    tree = et.parse(xmlname)
    root=tree.getroot()
    a=range(1,2**inp.index.stop-1)
    alltracers=d2b(a,inp.index.stop)
    #print(df2)
    #print(df2[sheets[0]][0][0])
    #print(df2[sheets[1]].loc[0:,1:3])
    #df2[sheets[1]].index
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
    #print(mat)
    #print(inpmetind)
    vec=np.delete(mat,inpmetind,0).reshape(-1)
    #print(vec)
    #print("pro_di:" + aa, flush=True)

   # remove any existing "result_xxxx.csv" file before create a new one.
    for file in glob.glob("result_[0-9]*-[0-9]*.csv"):
        os.remove(file)

    # create an output file(result_xxxx.csv) 
    with open(output_filename, 'w') as outfile:
        print(vec, file=outfile)
        #outfile.write(vec)
    outfile.close() 

    # clean-up; remove the uploaded input file
    if os.path.exists(xlsname):
        os.remove(xlsname)

    os.chdir(home_dir)
    return output_filename
    
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