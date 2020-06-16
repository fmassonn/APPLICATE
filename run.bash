#!/bin/bash -l
set -o nounset
set -o errexit
#set -x

module load Python/3.6.1-intel-2018

cd $HOME/git/APPLICATE/

outdir=/elic/web/fmasson/sea-ice-forecasts/

/opt/software/Python/3.6.1-intel-2018/bin/python ./APPLICATE-benchmark_SIO.py

chmod 744 out.html current_outlook.png current_pdf.png
mkdir -p $outdir/oper
rsync -rlptoDv current_outlook.png current_pdf.png out.html $outdir/oper

