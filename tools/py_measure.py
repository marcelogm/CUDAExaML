#!/usr/bin/python
import sys

class LogItem(object):
    execution = 0
    time = 0
        
def analyse(f):
    dct = dict()
    for line in f:
        splited = line.split(";", -1)
        if splited[0] in dct:
            dct[splited[0]].execution += 1
            dct[splited[0]].time = dct[splited[0]].time  + int(float(splited[1]))
        else:
            dct[splited[0]] = LogItem()
            dct[splited[0]].execution = 1
            dct[splited[0]].time = int(float(splited[1]))
    for key, value in dct.items():
        print "{};{};{}".format(key, value.time, value.execution)

def read():
    try:
        param = sys.argv[1]
        f = open(param, 'r')
        with f:
            analyse(f)
    except IOError:
        print 'Nao foi possivel abrir o arquivo.'

if len(sys.argv) <= 1:
    print "Envie o nome do arquivo por parametro"
else:
    read()