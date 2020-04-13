from setuptools import setup

setup(
      name='stackoverflow_predict', 
      version='1.0', 
      include_package_data=True, 
      scripts=["data/processdata.py","prediction/predictionclass.py"]
      )
    

