# pyPSSGA
Power System Simulation with PSSE

        Author:MD. Nazmuddoha Ansary, Sajjad Uddin Mahmud
        version:0.0.2

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

# SCR Calculation
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
                "SC_SIZE_MIN"   : 10,
                "SC_SIZE_MAX"   : 50,
                "STEP_SIZE"     : 5,
                "THRESH"        : 3.00
        }

* execute: ``` optimize.py```
> change dir if necessary

* Example Execution results

![](/src_img/exec.PNG?raw=true)

# Case Diagram

![](/src_img/main.jpg?raw=true)
