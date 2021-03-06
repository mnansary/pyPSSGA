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
from itertools import combinations
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
config_data     =   readJson('config.json')
CASE_STUDY      =   config_data['CASE_PATH']
BUS_IDS         =   str(config_data['BUS_IDS']).split(',')
BUS_IDS         =   [int(_id) for _id in BUS_IDS ]
MACHINE_IDS     =   str(config_data['GEN_IDS']).split(',')
MACHINE_IDS     =   [int(_id) for _id in MACHINE_IDS]
SC_IDS          =   str(config_data['SC_IDS']).split(',')
SC_IDS          =   [int(_id) for _id in SC_IDS]
SC_SIZE_MIN     =   config_data['SC_SIZE_MIN']
SC_SIZE_MAX     =   config_data['SC_SIZE_MAX']
CROSS_OVER      =   config_data['CROSS_OVER']
THRESH          =   config_data['THRESH']
POP_SIZE        =   config_data['POP_SIZE']
MAX_ITER        =   config_data['MAX_ITER']
MUTATION_RATE   =   config_data["MUTATION_RATE"]
MUTATION_LIMIT  =   int(MUTATION_RATE*SC_SIZE_MAX) 
TERM_CHECK      =   config_data["TERM_CHECK"] 
# cross over params
N_CROSS         =   int(CROSS_OVER*len(SC_IDS))
CROSS_LIST      =   [i for i in range(len(SC_IDS))]
# NPV params
C_f             =   config_data['FIXED_COST']
C_v             =   config_data['VARIABLE_COST']
C_m             =   config_data['SERVICE_COST']
C_MWh           =   config_data['MWH_COST']
C_fuel          =   config_data['FUEL_COST']
P_loss          =   config_data['PWR_LOSS_PERC']/100.0
Tsc             =   config_data['ON_TIME_PERC']/100.0
alpha           =   config_data['CURTAIL_PERC']/100.0
r               =   config_data['DISSCOUNT_PERC']/100.0
N_years         =   config_data['NUM_OF_YEARS']
P_gen           =   config_data['PWR_GEN_FACTOR']

C_MWh           =   C_MWh[:N_years]
C_fuel          =   C_fuel[:N_years]

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
def __verboseGen(i,best_member,best_fitness):
    print(colored('gen:{}'.format(i),color='yellow'),'|',
        colored('best_parent:{}'.format(best_member),color='green'),'|',
        colored('best_fitness:{}'.format(best_fitness),color='green'))
        
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
        _X=np.random.randint(low=SC_SIZE_MIN,high=SC_SIZE_MAX,size=(1000,len(SC_IDS)))
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
        C       =   [] # cost per year
        # calculate C0,Cmain,Celec
        for idx in range(len(x)):
            C0      +=  C_f+C_v*x[idx]
            Cmain   +=  C_m*x[idx]

        for c_mwh in C_MWh:
            Celec=0
            for idx in range(len(x)):
                Celec   +=  365*24*Tsc*c_mwh*P_loss*x[idx]
            C.append(Celec+Cmain)
        # revenue
        R=[]
        for c_mwh,c_fuel in zip(C_MWh,C_fuel):
            Rev=0
            for Pgen in PWR_GEN:
                Rev       +=  365*24*alpha*Pgen*(c_mwh+c_fuel)
            R.append(Rev)

        # NPV
        npv=0
        for i in range(1,N_years+1):
            npv     +=  ((R[i-1]-C[i-1])/(math.pow((1+r),i)))-C0
        NPV.append(math.ceil(npv))
    return np.asarray(NPV)
# ---
def get_fitest(population):
    population=np.array(population)
    _fitness=fitness_function(population)
    Fitness=list(zip(population,_fitness))
    Fitness.sort(key= lambda x:x[1],reverse=True) # sort based on fitness
    best_member,best_fitness=Fitness[0]
    population,_=zip(*Fitness)
    population=np.array(population[:POP_SIZE])
    return population,best_member,best_fitness
# ---
def getChildren(s1,s2):
    idxs=random.sample(CROSS_LIST,N_CROSS)
    s1[idxs],s2[idxs]=s2[idxs],s1[idxs]
    return s1,s2

def corssOver(parents):
    children=[]
    for s1,s2 in combinations(parents, 2):
        c1,c2=getChildren(s1,s2)
        children.append(c1)
        children.append(c2)
    return children
    
# ---
def mutate(children):
    fit_children=[]
    children=children+np.random.randint(low=-MUTATION_LIMIT,high=MUTATION_LIMIT,size=np.array(children).shape)
    for x in tqdm(children):
        if all(val >=SC_SIZE_MIN and val <=SC_SIZE_MAX  for val in x):
            if check_constraint(x):
                fit_children.append(x)
    return fit_children
# ---    
def GA(parents,max_iter,verbose=True):
    LOG_INFO('Running Iterations')
    parents,best_member,best_fitness=get_fitest(parents)
    i=0
    not_improving=0
    par=[]
    fit=[]
    par.append(best_member)
    fit.append(best_fitness)
    if verbose:
        __verboseGen(i,best_member,best_fitness)

    for i in range(1,max_iter):
        children=corssOver(parents)
        children=mutate(children)
        if not children:
            LOG_INFO('TERMINATION CONDITION REACHED(NO FIT CHILDREN)',pcolor='red')
            break
        else:
            parents,curr_best_member,curr_best_fitness=get_fitest(children)
            if  curr_best_fitness > best_fitness:
                if not_improving !=0:
                    not_improving=0
                best_fitness = curr_best_fitness
                best_member  = curr_best_member
            else:
                not_improving+=1
                if verbose:
                    LOG_INFO('best fitness has not increased for {} iterations'.format(not_improving),pcolor='red')
                if not_improving==TERM_CHECK:
                    LOG_INFO('TERMINATION CONDITION REACHED(NPV CONVERGENCE WITHIN 10 ITERATIONS)',pcolor='red')
                    break
            if verbose:
                __verboseGen(i,best_member,best_fitness)
            par.append(best_member)
            fit.append(best_fitness)
            
    return best_member,par,fit



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