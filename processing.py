import statistics
import os
import matlab.engine
from datetime import datetime
import glob

def get_mode(number_list):
    try:
        return "The mode of the numbers is {}".format(statistics.mode(number_list))
    except statistics.StatisticsError as exc:
        return "Error calculating mode: {}".format(exc)

def process_file(input_file):
    output_var = ["mat", "vec", "metind", "expind"]
    output_filename = f'result_{str(datetime.now().strftime("%Y%m%d-%H%M%S"))}.csv'
    home_dir = os.getcwd()
    os.chdir(home_dir + "/app/matlab")
    #print(os.getcwd(), flush=True)
    eng = matlab.engine.start_matlab()
    input_data = input_file.stream.read().decode("utf-8")
    for line in input_data.splitlines():
        eng.eval(line, nargout=0)
    eng.simple1(nargout=0)

    # clean-up; remove all "result_xxxxx.csv" file
    for file in glob.glob("result_[0-9]*-[0-9]*.csv"):
        os.remove(file)
        
    # create "result_xxxx.csv" file by concatenate all 4 output files crated by simple1.m (matlab script) 
    with open(output_filename, 'w') as outfile:
        for o in output_var:
            if os.path.exists(o + ".txt"):
                with open(o + ".txt", 'r') as infile:
                    outfile.write(o + '\n')
                    for line in infile:
                        outfile.write(line)
                infile.close()
    outfile.close() 

    # clean-up; remove all 4 matlab output files
    for o in output_var:
        if os.path.exists(o + ".txt"):
            os.remove(o + ".txt")

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
