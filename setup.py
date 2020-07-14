from setuptools import setup, find_packages

import sys
setup(
        name='IntelliMaintLite',
		version='0.0.8',
        packages=find_packages(),
        install_requires=['sklearn', 'GPy', 'minisom', "numpy == 1.18.1",\
		'scipy==1.4.1', 'matplotlib', 'mplcursors',\
		'celluloid', 'PyWavelets', 'EMD-signal', \
		'pandas', 'scikit-learn',  'gast',\
		'seaborn', 'keras', 'tensorflow','pickle-mixin','mlxtend'], #removed pandas profiling
        
		author='IntelliMaint Data-Analytics Team',
		author_email="iptgithub@intellipredikt.com",
		description="IntelliMaint: A prognostics library for time series data by IPT",
        
        zip_safe=True,
        include_package_data=True        
		#python_requires='>=3.6',
	)
