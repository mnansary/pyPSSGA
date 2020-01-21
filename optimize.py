 # -*- coding: utf-8 -*-
'''
@author: MD. Nazmuddoha Ansary, Sajjad Uddin Mahmud
'''
from __future__ import print_function
import pssexplore34
import psspy
import pssarrays
import redirect
import sys
import os
import numpy as np
import pandas as pd
import math
import json
import itertools
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm 
import contextlib
from termcolor import colored
os.system('color')
#-----------------------------------------------------------------------------------------------------------
redirect.psse2py()
psspy.throwPsseExceptions=True
OLD_STDOUT = sys.stdout  
@contextlib.contextmanager  
def silence(new_target=None):
    if new_target is None:
        new_target = open(os.devnull, 'w')
    old_target, sys.stdout = sys.stdout, new_target # replace sys.stdout
    try:
        yield new_target # run some code with the replaced stdout
    finally:
        sys.stdout = old_target # restore to the previous value
#-----------------------------------------------------------------------------------------------------------
def LOG_INFO(msg,pcolor='green'):
    print(colored('#LOG:    ',color='blue'),colored(msg,color=pcolor))

def readJson(file_name):
    return json.load(open(file_name))

def banner():
    _stars  ='***********************************************'
    _title  ='*   Synchronous Condenser Size Optimization   *'
    _method ='*         Method: Genetic Algorithm           *'
    _auth_0 ='*                 Authors                     *'  
    _auth_1 ='*          MD. Nazmuddoha Ansary              *'  
    _auth_2 ='*           Sajjad Uddin Mahmud               *'  
    print(colored(_stars,color='yellow'))
    print(colored(_title,color='yellow'))
    print(colored(_method,color='yellow'))
    print(colored(_auth_0,color='yellow'))
    print(colored(_auth_1,color='yellow'))
    print(colored(_auth_2,color='yellow'))
    print(colored(_stars,color='yellow'))

#-----------------------------------------------------------------------------------------------------------
LOG_INFO('Loading System Config ')
config_data =   readJson('config.json')
CASE_STUDY  =   config_data['CASE_PATH']
BUS_IDS     =   str(config_data['BUS_IDS']).split(',')
BUS_IDS     =   [int(_id) for _id in BUS_IDS ]
MACHINE_IDS =   str(config_data['GEN_IDS']).split(',')
MACHINE_IDS =   [int(_id) for _id in MACHINE_IDS]
SC_IDS      =   str(config_data['SC_IDS']).split(',')
SC_IDS      =   [int(_id) for _id in SC_IDS]
SC_SIZE_MIN =   config_data['SC_SIZE_MIN']
SC_SIZE_MAX =   config_data['SC_SIZE_MAX']
STEP_SIZE   =   config_data['STEP_SIZE']
THRESH      =   config_data['THRESH']
POP_SIZE    =   config_data['POP_SIZE']
MAX_ITER    =   config_data['MAX_ITER']

INITIAL_SIZES=str(config_data['INITIAL_SIZES']).split(',')
INITIAL_SIZES=[int(_size) for _size in INITIAL_SIZES ]
INITIAL_SIZES=np.asarray(INITIAL_SIZES)
#-----------------------------------------------------------------------------------------------------------
def initCase():
    LOG_INFO('Initializing psspy')
    with silence():
        # initialize
        psspy.psseinit(8000)
    LOG_INFO('Loading Case ')
    with silence():
        # load case
        psspy.case(CASE_STUDY)
#-----------------------------------------------------------------------------------------------------------
def __summary(hv_bus_ids,gen_bus_ids,pmax,i_sym,v_kv,v_pu,scr,save_csv=True):
    # intialise data of lists. 
    data = {'HV BUS':hv_bus_ids, 
            'GEN BUS':gen_bus_ids,
            'Pmax (MW)':pmax,
            'I"(A)':i_sym,
            'Voltage (kv)':v_kv,
            'p.u(V)':v_pu}
    # Create DataFrame 
    df = pd.DataFrame(data)
    df['SCR']=scr
    LOG_INFO('Summary:') 
    print(colored(df.head(len(hv_bus_ids)),color='yellow'))
    if save_csv:
        _csv_path=os.path.join(os.getcwd(),'summary.csv')
        df.to_csv(_csv_path)
        LOG_INFO('Saved Summary at: {}'.format(_csv_path))
#-----------------------------------------------------------------------------------------------------------
def modifyCaseBySize(sizes):
    for idx in range(len(SC_IDS)):
        with silence():
            psspy.machine_data_2(i=SC_IDS[idx],
                                 id='1',
                                 intgar=[1,0,0,0,0,0],
                                 realar=[0,0,500,-200,0,0,sizes[idx],0,0.09,0,0,1,1,1,1,1,1])
#-----------------------------------------------------------------------------------------------------------
OPTIM_SIZE=None
def checkOptim(sc,scr,TOTAL_MIN):
    global OPTIM_SIZE
    if (all(val > THRESH for val in scr) and sum(list(sc))<TOTAL_MIN):
        TOTAL_MIN =sum(list(sc))
        OPTIM_SIZE=np.asarray(sc)
    return TOTAL_MIN

def optimMan():
    __sizes=range(SC_SIZE_MIN,SC_SIZE_MAX+STEP_SIZE,STEP_SIZE)
    _list_sizes=[__sizes for _ in range(len(SC_IDS))]
    combs=list(itertools.product(*_list_sizes))
    size_vals=[]
    scr_vals=[]
    LOG_INFO('Calculating Initial Inclusive Sizes')
    TOTAL_MIN   =   SC_SIZE_MAX*len(SC_IDS)
    for sc in tqdm(combs):
        modifyCaseBySize(sc)
        scr=calcSCR(hv_bus_ids=BUS_IDS,gen_bus_ids=MACHINE_IDS)
        size_vals.append(list(sc))
        scr_vals.append(scr)
        TOTAL_MIN=checkOptim(sc,scr,TOTAL_MIN)
    
def sizeSummary():
    LOG_INFO('Sizes for Sync. Condensors',pcolor='yellow')
    for idx in range(len(SC_IDS)):
        print(colored('Sync. Cond. No.: {}        Size:  {}'.format(SC_IDS[idx],OPTIM_SIZE[idx]),color='yellow'))
#-----------------------------------------------------------------------------------------------------------
def calcSCR(hv_bus_ids,gen_bus_ids,ret_params=False):
    '''
    ARGS: 
        hv_bus_ids  : list of relevant high voltage buses <LIST OF INT>
        gen_bus_ids : list of relevant generator buses <LIST OF INT>
   RETURNS:
        scr         : list of short circuit ratios
    Calculation:
        scr=sqrt(3)*i_sym*v_pu*v_kv*_UNIT_FACTOR/pmax
        _UNIT_FACTOR=1e-3
    '''
    # params
    pmax=[]
    i_sym=[]
    v_kv=[]
    v_pu=[]
    # non-verbose execution
    #LOG_INFO('Solving for full newton-rafsan ')
    with silence():
        # full newton-rafsan
        psspy.fnsl(options4=1,
                   options5=1,
                   options6=1)
    #LOG_INFO('Getting IECS values')
    with silence():
        # get symmetric 3-phase fault currents
        all_currents=pssarrays.iecs_currents(all=1,
                                            flt3ph=1,
                                            optnftrc=2,
                                            vfactorc=1.0)
    # all bus pu voltage
    _, (__v_pu,) = psspy.abusreal(sid=-1, string=["PU"])
    # all bus kv voltage
    _, (__v_kv,) = psspy.abusreal(sid=-1, string=["BASE"])
    # get pmax
    for _id in gen_bus_ids:
        _, _pmax = psspy.macdat(ibus=_id, id='1', string='PMAX') # find power of machine
        pmax.append(_pmax)
    # get v_pu,v_kv,i_sym
    for _id in hv_bus_ids:
        v_pu.append(__v_pu[_id-1])
        v_kv.append(__v_kv[_id-1])
        i_sym.append(all_currents.flt3ph[_id-1].ibsym.real)
    
    total_bus=len(pmax)
    scr=[]
    _UNIT_FACTOR=1e-3
    #LOG_INFO('Calculating SCR')
    for idx in range(total_bus):
        scr.append(math.sqrt(3)*i_sym[idx]*v_pu[idx]*v_kv[idx]*_UNIT_FACTOR/pmax[idx])
    
    if ret_params:
        return pmax,i_sym,v_kv,v_pu,scr
    else:
        return scr
    
#-----------------------------------------------------------------------------------------------------------
# ---
def fitness_function(X):
    Y=[]
    for x in X:
        modifyCaseBySize(x)
        scr=calcSCR(hv_bus_ids=BUS_IDS,gen_bus_ids=MACHINE_IDS)
        if all(val > THRESH for val in scr):
            y=np.sum(x)
        else:
            y=len(SC_IDS)*SC_SIZE_MAX

        Y.append(y)
    return np.asarray(Y)
# ---
def mutate(parents):
    n=len(parents)
    scores=fitness_function(parents)
    #idx=scores < len(SC_IDS)*SC_SIZE_MAX
    idx=scores > 0
    scores=scores[idx]
    parents=np.array(parents)[idx]
    children=parents[np.random.choice(parents.shape[0],size=n,p=scores.astype('float32')/np.sum(scores))]
    children=children+np.random.randint(low=-STEP_SIZE,high=STEP_SIZE,size=children.shape)
    return children
# ---
def get_fitest_parents(parents):
    _fitness=fitness_function(parents)
    PFitness=list(zip(parents,_fitness))
    PFitness.sort(key= lambda x:x[1]) # sort based on fitness
    best_parent,best_fitness=PFitness[0]
    return best_parent,best_fitness
# ---    
def GA(parents,max_iter):
    LOG_INFO('Running Iterations')
    global OPTIM_SIZE
    best_parent,best_fitness=get_fitest_parents(parents)
    PARENTS=[best_parent]
    FITNESS=[best_fitness]
    for i in tqdm(range(1,max_iter)):
        parents=mutate(parents)
        curr_parent,curr_fitness=get_fitest_parents(parents)
        if curr_fitness < best_fitness:
            best_fitness = curr_fitness
            best_parent  = curr_parent
        PARENTS.append(best_parent)
        FITNESS.append(best_fitness)
    OPTIM_SIZE=best_parent
    return PARENTS,FITNESS
    
#-----------------------------------------------------------------------------------------------------------
def main():
    LOG_INFO('Initializing Population')
    X=np.random.randint(low=SC_SIZE_MIN,high=SC_SIZE_MAX,size=(POP_SIZE-1,len(SC_IDS)))
    X=np.vstack((X,INITIAL_SIZES))
    sizes,totals=GA(X,MAX_ITER)
    sizeSummary()
    col_names=['Size_{}'.format(SC_IDS[i]) for i in range(len(SC_IDS))]
    col_names+=['Total Size']
    SIZES=np.vstack(sizes)
    TOTALS=np.vstack(totals)
    DATA=np.concatenate((SIZES,TOTALS),axis=1)   
    df = pd.DataFrame(data=DATA,columns=col_names)
    _csv_path=os.path.join(os.getcwd(),'history.csv')
    df.to_csv(_csv_path)
    df.plot()
    plt.show()
#-----------------------------------------------------------------------------------------------------------
if __name__=='__main__':
    start_time=time()
    banner()
    initCase()
    main()
    LOG_INFO('Time Taken:{} sec'.format(time()-start_time),pcolor='cyan')