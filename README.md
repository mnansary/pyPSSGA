# pyPSSGA
Power System Simulation with PSSE

        Author:MD. Nazmuddoha Ansary, Sajjad Uddin Mahmud
        version:0.0.4

# Environment

* PSSEXplore:**340302**
* python: **2.7.13**
* pip: **9.0.1**
* ```pip install -r requirements.txt```

        astroid==1.6.6
        backports.functools-lru-cache==1.6.1
        colorama==0.4.3
        configparser==4.0.2
        cycler==0.10.0
        enum34==1.1.6
        futures==3.3.0
        isort==4.3.21
        kiwisolver==1.1.0
        lazy-object-proxy==1.4.3
        matplotlib==2.2.4
        mccabe==0.6.1
        numpy==1.16.6
        pandas==0.24.2
        pylint==1.9.5
        pyparsing==2.4.6
        python-dateutil==2.8.1
        pytz==2019.3
        singledispatch==3.4.0.3
        six==1.14.0
        termcolor==1.1.0
        tqdm==4.41.1
        wrapt==1.11.2

# Setup
* Add **python27** binary path to **environment variables**
* Create a **_some_name_.pth** (*eg:_psse_.pth*) file in **Site-packages** dir of python installation dir *example:C:\Python27\Lib\site-packages*
* **_psse_.pth** should contain: **C:\Program Files (x86)\PTI\PSSEXplore34\PSSPY27** 

# NPV optimization
* Edit values in **config.json**
    * CASE_PATH   : path to case file (.sav)
    * BUS_IDS     : list of relevant high voltage buses 
    * GEN_IDS     : list of relevant generator buses
    * SC_IDS      : list of relevant condensor buses 

* example *config.json*:

        {
            "CASE_PATH"     :"F:\\PSSE\\code\\pyPSSGA\\base_case_final.sav",
            "BUS_IDS"       :"14,15,16,20",
            "GEN_IDS"       :"30,36,32,34",
            "SC_IDS"        :"37,38,39,40",
            "SC_SIZE_MIN"   : 5,
            "SC_SIZE_MAX"   : 100,
            "STEP_SIZE"     : 5,
            "THRESH"        : 3,
            "POP_SIZE"      : 25,
            "MAX_ITER"      : 500,
            "FIXED_COST"    : 1000000,
            "VARIABLE_COST" : 30000,
            "SERVICE_COST"  : 800,
            "FUEL_COST"     : 100, 
            "MWH_COST"      : 100,
            "PWR_LOSS_PERC" : 3,
            "ON_TIME_PERC"  : 100,
            "CURTAIL_PERC"  : 1,
            "DISSCOUNT_PERC": 7,
            "NUM_OF_YEARS"  : 20,
            "PWR_GEN_FACTOR": 0.8
        }

* execute: ``` optimize.py```
> change dir if necessary

* Example Execution results

![](/src_img/exec.PNG?raw=true)

# Summary

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GEN BUS</th>
      <th>HV BUS</th>
      <th>I"(A)</th>
      <th>Pmax (MW)</th>
      <th>Voltage (kv)</th>
      <th>p.u(V)</th>
      <th>SCR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>30</td>
      <td>14</td>
      <td>5293.145996</td>
      <td>1050.000000</td>
      <td>345.0</td>
      <td>0.997127</td>
      <td>3.003686</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36</td>
      <td>15</td>
      <td>3640.414307</td>
      <td>720.000000</td>
      <td>345.0</td>
      <td>0.995584</td>
      <td>3.007988</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>16</td>
      <td>4843.773438</td>
      <td>960.000061</td>
      <td>345.0</td>
      <td>0.997276</td>
      <td>3.006821</td>
    </tr>
    <tr>
      <th>3</th>
      <td>34</td>
      <td>20</td>
      <td>4295.235352</td>
      <td>839.999939</td>
      <td>345.0</td>
      <td>0.984831</td>
      <td>3.009187</td>
    </tr>
  </tbody>
</table>

# History

![](/src_img/history_sizes.png?raw=true)

![](/src_img/history_SCR.png?raw=true)

![](/src_img/history_NPV.png?raw=true)

# Case Diagram

![](/src_img/main.jpg?raw=true)
