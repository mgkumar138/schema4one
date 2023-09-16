# Requirements to run demo
1) Open bash or terminal

2) Create & activate Conda environment 
```
$ conda create -n s4o python=3.7 numpy scipy pandas matplotlib jupyter
$ conda activate s4o
```

3) Install tensorflow (GPU not necessary)
```
$ python -m pip install tensorflow=2.5
```

4) Start Jupyter notebook IDE
```
jupyter notebook
```

5) Navigate to "examples" directory and open bioplaus_oneshot_multiPA_demo.ipynb to run.

6) Training the agent from scratch on the MPA task will take approximately 1 hour, and another 1 hour on the NM task. 