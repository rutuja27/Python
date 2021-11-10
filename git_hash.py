# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 17:00:32 2021

@author: 27rut
"""

import subprocess
from datetime import datetime
import os
import pprint
import subprocess

 
def get_sha():
   
    BIAS_path = 'C:/Users/27rut/BIAS/BIASJAABA'
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=BIAS_path).decode('ascii').strip()
    return sha[0:5]

#dir is not keyword
def makemydir(path_dir):
    
  path_dir += '/' + str(get_sha()) + '_' + get_current_date()
  print(path_dir)  
  try:
    os.makedirs(path_dir)
  except OSError:
    pass
  # let exception propagate if we just can't
  # cd into the specified directory
  os.chdir(path_dir)
  
def get_current_date():
    current_datetime = datetime.now()
    return  str(current_datetime.month) + \
            '_' + str(current_datetime.day) + '_' + \
            str(current_datetime.year)
    
