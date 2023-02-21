import statistics

def get_mode(number_list):
    try:
        return "The mode of the numbers is {}".format(statistics.mode(number_list))
    except statistics.StatisticsError as exc:
        return "Error calculating mode: {}".format(exc)

def process_data(input_data):
    result = ""
    #for line in input_data.splitlines():
    #    if line != "":
    #        numbers = [float(n.strip()) for n in line.split(",")]
    #        result += str(sum(numbers))
    #    result += "\n"
    return result

def do_addition(number1, number2):
    return number1 + number2