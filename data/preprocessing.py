import os
import sys

def CSVconverter(file_name):
    label_output = open(file_name+".label.csv", 'w')
    data_output = open(file_name+".data.csv", 'w')
    feat_size = [4389, 8000, 329, 7432, 2646, 428, 233, 6301, 295, 11, 173, 176642,\
                585, 147117, 19845, 14830, 6916, 18687, 4, 6646, 1272, 46, 141085, 64381,\
                63692, 11, 2156, 7806, 61, 5, 928, 15, 147387, 116331, 145634, 57186, 9307, 63, 34]
    with open(file_name) as _file_:
        line = _file_.readline()
        while line:
            label_output.write(line[0]+'\n')
            line = line.split(',')[1:]
            string = [str(hash(line[j])%feat_size[j]) for j in range(len(line))]
            string = ",".join(string)
            data_output.write(string+'\n')
            line = _file_.readline()

if __name__ == "__main__":
    CSVconverter(sys.argv[1])

