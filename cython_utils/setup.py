from distutils.core import setup
from distutils.extension import Extension
import sys

sys.path.append('/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')

from Cython.Build import cythonize
import numpy
import os

if os.name =='nt' :

    ext_modules=[
        Extension("nms",
                sources=["nms.pyx"],
                #libraries=["m"] # Unix-like specific
                    include_dirs=[numpy.get_include()]
        )
    ]

    ext_modules_yolo2=[
        Extension("cy_yolo2_findboxes",
                  sources=["cy_yolo2_findboxes.pyx"],
                  #libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()]
        )
    ]

    ext_modules_yolo=[
        Extension("cy_yolo_findboxes",
                  sources=["cy_yolo_findboxes.pyx"],
                  #libraries=["m"] # Unix-like specific
                  include_dirs=[numpy.get_include()]
        )
    ]

else :

    ext_modules=[
        Extension("nms",
                sources=["nms.pyx"],
                libraries=["m"], # Unix-like specific
                include_dirs=[numpy.get_include()]
        )
    ]

    ext_modules_yolo2=[
        Extension("cy_yolo2_findboxes",
                  sources=["cy_yolo2_findboxes.pyx"],
                  libraries=["m"],# Unix-like specific
                  include_dirs=[numpy.get_include()]
        )
    ]

    ext_modules_yolo=[
        Extension("cy_yolo_findboxes",
                  sources=["cy_yolo_findboxes.pyx"],
                  libraries=["m"], # Unix-like specific
                  include_dirs=[numpy.get_include()]
        )
    ]




setup(

    #name= 'cy_findboxes',
    ext_modules = cythonize(ext_modules),
)

setup(

    #name= 'cy_findboxes',
    ext_modules = cythonize(ext_modules_yolo2),
)



setup(

    #name= 'cy_findboxes',
    ext_modules = cythonize(ext_modules_yolo),
)
