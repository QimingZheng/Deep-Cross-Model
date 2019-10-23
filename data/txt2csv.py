import os
import sys

def txt2csv(file_name):
    output = open(file_name[:-4]+".csv", 'w')
    with open(file_name) as _file_:
        line = _file_.readline()
        while(line):
            output.write(line.replace('\t', ','))
            line = _file_.readline()

if __name__ == "__main__":
    txt2csv(sys.argv[1])
