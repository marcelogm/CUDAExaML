#!/usr/bin/python
import sys, random, string, os

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(string.letters + string.digits) for i in range(length))

def getSize(filename):
    st = os.stat(filename)
    return st.st_size

def writeTaxa(f,name, phrase):
    f.write(name + "\n")
    f.write(phrase)
    f.write("\n")

def split(f):
    taxa = int(sys.argv[2])
    patterns = int(sys.argv[3])
    n = open("{}_{}".format(taxa, patterns), "a+")
    n.write(" {} {}".format(taxa, patterns))
    for x in range(0, taxa):
	p = f.read(patterns)
	if(p and len(p) == patterns):
       	    writeTaxa(n, randomword(10), p)
	else:
	    print("Arquivo muito pequeno para isso")
	    exit()
    n.close()
    f.close()

def read():
    try:
        param = sys.argv[1]
        f = open(param, 'r')
        with f:
            split(f)
    except IOError:
        print 'Nao foi possivel abrir o arquivo.'

if len(sys.argv) <= 3:
    print "Envie os parametros corretamente."
else:
    read()
