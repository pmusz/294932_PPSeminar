# Python_UMK_2020

usage: projektPM.py [-h] [-d D] [-i I] [-c C] [-a A]

Dimensionality reduction of tensor datasets.

optional arguments:
  -h, --help  show this help message and exit
  -d D        Decimation ratio for reducing amount of input data (5..10000).
  -i I        Input filename from molecular dynamics database on FU Berlin FTP
              server, otherwise a default example file will be used.
  -c C        Input name of desired colormap (ocean, viridis, gist_earth are
              supported).
  -a A        Input name of desired dimensionality reduction algorithm (PCA
              and TSNE are supported).
