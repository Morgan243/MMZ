from setuptools import setup, find_packages
setup(name='MMZ',
      version='0.1',
      description='Various models, mostly torch implementations',
      author='Morgan Stuart',
      packages=['mmz'],
      #packages=find_packages(),
      #modules=['feature_processing', 'torch_models'],
      requires=['numpy', 'pandas',
                'sklearn', 'torch', 'torchvision',
                'attrs'])