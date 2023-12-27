# shapedtw-timeseries

This study is part of a project for the course "Machine Learning for Time Series" in the master MVA, at ENS Paris-Saclay.

The goal is to implement, experiment in Python and extend some parts of the _shapeDTW: shape Dynamic Time Warping_ paper from Jiaping Zhao and Laurent Itti.

Part of the code is based on these two repositories : 
- https://github.com/jiapingz/shapeDTW
- https://github.com/MikolajSzafraniecUPDS/shapedtw-python


Here is the tree of the repository:
```bash
.
├── README.md
├── data
│   ├── results                 #results from experiment 2 (speech recognition)
│   └── savee                   #data for experiment 2 (speech recognition)
├── requirements.txt
├── shapedtw                    #folder with python files to compute the shapedtw (already implemented at https://github.com/MikolajSzafraniecUPDS/shapedtw-python)
│   ├── __init__.py
│   ├── dtwPlot.py
│   ├── exceptions.py
│   ├── preprocessing.py
│   ├── shapeDescriptors.py
│   ├── shapedtw.py
│   └── utils.py
├── experiments.ipynb           #main notebook to run with our experiments
├── simulated_timeseries.py     #python file with methods used in experiment 1
└── speech_recognition.py       #python file with methods used in experiment 2
```
