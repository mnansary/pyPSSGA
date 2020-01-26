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
import random
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
# NPV params
C_f         =   config_data['FIXED_COST']
C_v         =   config_data['VARIABLE_COST']
C_m         =   config_data['SERVICE_COST']
C_MWh       =   config_data['MWH_COST']
C_fuel      =   config_data['FUEL_COST']
P_loss      =   config_data['PWR_LOSS_PERC']/100.0
Tsc         =   config_data['ON_TIME_PERC']/100.0
alpha       =   config_data['CURTAIL_PERC']/100.0
r           =   config_data['DISSCOUNT_PERC']/100.0
N_years     =   config_data['NUM_OF_YEARS']
P_gen       =   config_data['PWR_GEN_FACTOR']
#-----------------------------------------------------------------------------------------------------------
def getGeneratedPower():
    with silence():
        # initialize
        psspy.psseinit()
        # load case
        psspy.case(CASE_STUDY)
    PWR_GEN=[]
    for _id in MACHINE_IDS:
            _, _pmax = psspy.macdat(ibus=_id, id='1', string='PMAX') # find power of machine
            Pgen    =   P_gen*_pmax
            PWR_GEN.append(Pgen)
    return PWR_GEN
PWR_GEN     =   getGeneratedPower()
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
        df.to_html('summary.html')
        LOG_INFO('Saved Summary at: {}'.format(_csv_path))

def sizeSummary(sizes):
    LOG_INFO('Sizes for Sync. Condensors',pcolor='yellow')
    for idx in range(len(SC_IDS)):
        print(colored('Sync. Cond. No.: {}        Size:  {}'.format(SC_IDS[idx],sizes[idx]),color='yellow'))

#-----------------------------------------------------------------------------------------------------------
def calcSCR(sizes,ret_params=False):
    '''
    ARGS: 
        sizes       : list of sizes of Sync. Conds  
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
    with silence():
        for idx in range(len(SC_IDS)):
            psspy.machine_data_2(i=SC_IDS[idx],
                                id='1',
                                intgar=[1,0,0,0,0,0],
                                realar=[0,0,500,-200,0,0,sizes[idx],0,0.17,0,0,1,1,1,1,1,1])
        # full newton-rafsan
        psspy.fnsl(options4=1,
                   options5=1,
                   options6=1)
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
    for _id in MACHINE_IDS:
        _, _pmax = psspy.macdat(ibus=_id, id='1', string='PMAX') # find power of machine
        pmax.append(_pmax)
    # get v_pu,v_kv,i_sym
    for _id in BUS_IDS:
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
def initPop():
    pop=0
    X=[]
    LOG_INFO('Initializing population')
    while (pop<POP_SIZE):
        _X=np.random.randint(low=SC_SIZE_MIN,high=SC_SIZE_MAX,size=(MAX_ITER,len(SC_IDS)))
        for x in tqdm(_X):
            if pop==POP_SIZE:
                break
            if check_constraint(x):
                X.append(x)
                pop+=1
    X=np.asarray(X)
    return X
    
   
# ---
def check_constraint(x):
    scr=calcSCR(x)
    if all(val > THRESH for val in scr):
        return True
    else:
        return False

# ---
def fitness_function(X):
    NPV=[]
    for x in X:
        C0      =   0
        Cmain   =   0
        Celec   =   0
        # calculate C0,Cmain,Celec
        for idx in range(len(x)):
            C0      +=  C_f+C_v*x[idx]
            Cmain   +=  C_m*x[idx]
            Celec   +=  365*24*Tsc*C_MWh*P_loss*x[idx]
        # per year Cost
        C=Celec+Cmain
        # revenue
        R=0
        for Pgen in PWR_GEN:
            R       +=  365*24*alpha*Pgen*(C_MWh+C_fuel)
        # NPV
        npv=0
        for i in range(1,N_years+1):
            #NPV    +=  ((math.pow(R,i)-math.pow(C,i))/(math.pow((1+r),i)))-C0
            npv     +=  ((R-C)/(math.pow((1+r),i)))-C0
        NPV.append(math.ceil(npv))
    return np.asarray(NPV)
# ---
def mutate(parents):
    fit_children=[]
    scores=fitness_function(parents)
    parents=np.array(parents)
    children=parents[np.random.choice(parents.shape[0],size=len(parents),p=scores.astype('float32')/np.sum(scores))]
    while (len(fit_children) < len(parents)):
        children=children+np.random.randint(low=-STEP_SIZE,high=STEP_SIZE,size=children.shape)
        for x in children:
            x[x<=0]=STEP_SIZE
            if len(fit_children)==len(parents):
                break
            if check_constraint(x):
                fit_children.append(x)
    fit_children=np.asarray(fit_children)
    return fit_children
            
# ---
def get_fitest_parents(parents):
    _fitness=fitness_function(parents)
    PFitness=list(zip(parents,_fitness))
    PFitness.sort(key= lambda x:x[1],reverse=True) # sort based on fitness
    best_parent,best_fitness=PFitness[0]
    return best_parent,best_fitness
# ---    
def GA(parents,max_iter,verbose=False):
    LOG_INFO('Running Iterations')
    curr_parent,curr_fitness=get_fitest_parents(parents)
    best_parent,best_fitness=curr_parent,curr_fitness
    i=0
    par=[]
    fit=[]
    par.append(best_parent)
    fit.append(best_fitness)
    if verbose:
        print(colored('gen:{}'.format(i),color='yellow'),'|',
            colored('best_parent:{}'.format(best_parent),color='green'),'|',
            colored('best_fitness:{}'.format(best_fitness),color='green'),'|',
            colored('curr_parent:{}'.format(curr_parent),color='blue'),'|',
            colored('curr_fitness:{}'.format(curr_fitness),color='blue'))
    for i in tqdm(range(1,max_iter)):
        
        parents=mutate(parents)
        curr_parent,curr_fitness=get_fitest_parents(parents)
        if  curr_fitness > best_fitness:
            best_fitness = curr_fitness
            best_parent  = curr_parent
        if i % 10 ==0 and verbose:
            print(colored('gen:{}'.format(i),color='yellow'),'|',
                colored('best_parent:{}'.format(best_parent),color='green'),'|',
                colored('best_fitness:{}'.format(best_fitness),color='green'),'|',
                colored('curr_parent:{}'.format(curr_parent),color='red'),'|',
                colored('curr_fitness:{}'.format(curr_fitness),color='red'))
        par.append(best_parent)
        fit.append(best_fitness)
    return best_parent,par,fit
    
#-----------------------------------------------------------------------------------------------------------
def saveOptimizationHistory(sizes,npvs):
    col_names=['Size_{}(MVA)'.format(SC_IDS[i]) for i in range(len(SC_IDS))]
    col_names+=['SCR_{}'.format(MACHINE_IDS[i]) for i in range(len(MACHINE_IDS))]
    col_names+=['Total Size(MVA)','NPV(in millions)']
    SIZES=np.vstack(sizes)
    NPVS=np.vstack(np.asarray(npvs)/C_f)
    SCRS=[]
    LOG_INFO('Saving History')
    for x in tqdm(sizes):
        SCRS.append(calcSCR(x))
    SCRS=np.vstack(SCRS)
    TOTALS=np.vstack(np.sum(SIZES,axis=1))
    DATA=np.concatenate((SIZES,SCRS,TOTALS,NPVS),axis=1)
    df = pd.DataFrame(data=DATA,columns=col_names)
    df.index.name = 'iterations'
    _csv_path=os.path.join(os.getcwd(),'history.csv')
    df.to_csv(_csv_path)
    df.to_html('history.html')
    df[['Size_{}(MVA)'.format(SC_IDS[i]) for i in range(len(SC_IDS))]].plot()
    plt.savefig(os.path.join(os.getcwd(),'src_img','history_sizes.png'),dpi=500)
    df[['SCR_{}'.format(MACHINE_IDS[i]) for i in range(len(MACHINE_IDS))]].plot()
    plt.savefig(os.path.join(os.getcwd(),'src_img','history_SCR.png'),dpi=500)
    df[['Total Size(MVA)','NPV(in millions)']].plot()
    plt.savefig(os.path.join(os.getcwd(),'src_img','history_NPV.png'),dpi=500)
    


def saveSummary(optim_size):
    pmax,i_sym,v_kv,v_pu,scr=calcSCR(optim_size,ret_params=True)
    __summary(BUS_IDS,MACHINE_IDS,pmax,i_sym,v_kv,v_pu,scr,save_csv=True)

def main():
    X=initPop()
    optim_size,sizes,npvs=GA(X,MAX_ITER)
    saveOptimizationHistory(sizes,npvs)
    sizeSummary(optim_size)
    saveSummary(optim_size)
    
#-----------------------------------------------------------------------------------------------------------
if __name__=='__main__':
    start_time=time()
    banner()
    main()
    LOG_INFO('Time Taken:{} sec'.format(time()-start_time),pcolor='cyan')