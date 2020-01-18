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
    _stars='***********************************************'
    _title='*              SCR Calculation                *'
    _auth ='* MD. Nazmuddoha Ansary , Sajjad Uddin Mahmud *'  
    print(colored(_stars,color='yellow'))
    print(colored(_title,color='yellow'))
    #print(colored(_auth,color='yellow'))
    print(colored(_stars,color='yellow'))

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
LOG_INFO('Loading System Config ')
config_data=readJson('config.json')

CASE_STUDY=config_data['CASE_PATH']
BUS_IDS=str(config_data['BUS_IDS']).split(',')
BUS_IDS=[int(_id) for _id in BUS_IDS ]

MACHINE_IDS=str(config_data['GEN_IDS']).split(',')
MACHINE_IDS=[int(_id) for _id in MACHINE_IDS]

SC_IDS=str(config_data['SC_IDS']).split(',')
SC_IDS=[int(_id) for _id in SC_IDS]
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

def modifyCase():
    __sizes=range(10,300,10)
    scrs=[]
    LOG_INFO('Calculating SCR')
    for __size in tqdm(__sizes):
        with silence():
            psspy.machine_data_2(i=SC_IDS[0],
                                id='1',
                                intgar=[1,0,0,0,0,0],
                                realar=[0,0,500,-200,0,0,__size,0,0.09,0,0,1,1,1,1,1,1])
        scr=calcSCR(hv_bus_ids=BUS_IDS,gen_bus_ids=MACHINE_IDS)
        scrs.append(scr[0])
    
    data = {'size':__sizes, 
            'scr':scrs}
    # Create DataFrame 
    df = pd.DataFrame(data)
    df.set_index('size',inplace=True)
    print(df.head())
    df.plot()
    plt.show()

def modifyCase4():
    __sizes=range(10,60,10)
    __iter_len=int(math.pow(len(__sizes),4)) 
    S1s=[]
    S2s=[]
    S3s=[]
    S4s=[]
    SCR1=[]
    SCR2=[]
    SCR3=[]
    SCR4=[]
    
    LOG_INFO('Calculating SCR')
    for s1,s2,s3,s4 in tqdm(itertools.product(__sizes,__sizes,__sizes,__sizes),total=__iter_len):
        
        with silence():
            psspy.machine_data_2(i=SC_IDS[0],
                                id='1',
                                intgar=[1,0,0,0,0,0],
                                realar=[0,0,500,-200,0,0,s1,0,0.09,0,0,1,1,1,1,1,1])

            psspy.machine_data_2(i=SC_IDS[1],
                                id='1',
                                intgar=[1,0,0,0,0,0],
                                realar=[0,0,500,-200,0,0,s2,0,0.09,0,0,1,1,1,1,1,1])

            psspy.machine_data_2(i=SC_IDS[2],
                                id='1',
                                intgar=[1,0,0,0,0,0],
                                realar=[0,0,500,-200,0,0,s3,0,0.09,0,0,1,1,1,1,1,1])

            psspy.machine_data_2(i=SC_IDS[3],
                                id='1',
                                intgar=[1,0,0,0,0,0],
                                realar=[0,0,500,-200,0,0,s4,0,0.09,0,0,1,1,1,1,1,1])

        scr=calcSCR(hv_bus_ids=BUS_IDS,gen_bus_ids=MACHINE_IDS)
        S1s.append(s1)
        S2s.append(s2)
        S3s.append(s3)
        S4s.append(s4)
        SCR1.append(scr[0])
        SCR2.append(scr[1])
        SCR3.append(scr[2])
        SCR4.append(scr[3])
        __thresh=3.00
        if (scr[0]>=__thresh and scr[1]>=__thresh and scr[2]>=__thresh and scr[3]>=__thresh ):
            LOG_INFO('Sizes:{}->{},{}->{},{}->{},{}->{}'.format(SC_IDS[0],s1,
                                                                SC_IDS[1],s2,
                                                                SC_IDS[2],s3,
                                                                SC_IDS[3],s4))
            break
    
    data = {'size_{}'.format(SC_IDS[0]):S1s, 
            'scr_{}'.format(SC_IDS[0]):SCR1,
            'size_{}'.format(SC_IDS[1]):S2s, 
            'scr_{}'.format(SC_IDS[1]):SCR2,
            'size_{}'.format(SC_IDS[2]):S3s, 
            'scr_{}'.format(SC_IDS[2]):SCR3,
            'size_{}'.format(SC_IDS[3]):S4s, 
            'scr_{}'.format(SC_IDS[3]):SCR4}
    # Create DataFrame 
    df = pd.DataFrame(data)
    _csv_path=os.path.join(os.getcwd(),'SCR.csv')
    df.to_csv(_csv_path)
    


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
if __name__=='__main__':
    start_time=time()
    banner()
    initCase()
    modifyCase4()
    #__summary(BUS_IDS,MACHINE_IDS,pmax,i_sym,v_kv,v_pu,scr,save_csv=True)
    LOG_INFO('Time Taken:{}'.format(time()-start_time),pcolor='cyan')