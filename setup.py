from distutils.core import setup
setup(
  name = 'widelinears',       
  packages = ['widelinears'],
  version = '0.1',
  license='MIT',
  description = 'Parallel pytorch Neural Networks',   
  author = 'Joao Figueira',                  
  author_email = 'joaoperfig@gmail.com',   
  url = 'https://github.com/joaoperfig/WideLinears', 
  download_url = 'https://github.com/joaoperfig/WideLinears/archive/refs/tags/v_0.2.tar.gz',  
  keywords = ['pytorch', 'parallel', 'linear'], 
  install_requires=[          
          'torch',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',   
    'Intended Audience :: Developers',      
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',  
    'Programming Language :: Python :: 3',    
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)