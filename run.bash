#!/bin/bash -l
set -o nounset
set -o errexit
#set -x

module load Python/3.6.1-intel-2018

cd $HOME/git/APPLICATE/

outdir=/elic/web/fmasson/sea-ice-forecasts/

/opt/software/Python/3.6.1-intel-2018/bin/python ./APPLICATE-benchmark_SIO.py

mkdir -p $outdir

# Create oper page
echo "<!DOCTYPE html>
<html>
<body>

<h1> Probabilistic forecasts of September mean Arctic sea ice extent <br>
based on recalibrated damped anomaly persistence forecast
</h1>

<img src="operational_pdf.png" width=1000 >
<br>

<img src="operational_outlook.png" width=1000 >
<br>

<h2>
<a href="https://www.elic.ucl.ac.be/users/fmasson/sea-ice-forecasts/hindcasts.html">Hindcasts</a>
</h2>

<h2>
This probabilistic statistical forecasting system has been developed as a contribution to the 
<a href="https://www.arcus.org/sipn/sea-ice-outlook"> Sea Ice Outlook</a>
</h2> as part of the <a href="https://www.applicate.eu"> APPLICATE </a> Horizon 2020 Project

<h2>
More details on the method can be found <a href="https://www.elic.ucl.ac.be/users/fmasson/sea-ice-forecasts/APPLICATE-benchmark.pdf">here</a>
</h2>

</body>
</html>
" > webpages/oper.html

  

# Create hindcast landing page
echo "<!DOCTYPE html>
<html>
<body>

<h1> Probabilistic hindcasts of September mean Arctic sea ice extent <br>
based on recalibrated damped anomaly persistence forecast
</h1>

Hindcasts are done in operational conditions (not using future data)

" > webpages/hindcasts.html

# Create hindcast pages
nowyear=$(date +%Y)
for year in `seq 1994 $((nowyear - 1))`
do

  # Complete landing page
  echo "<a href="https://www.elic.ucl.ac.be/users/fmasson/sea-ice-forecasts/${year}.html">${year}</a>" >> webpages/hindcasts.html

  # Design year's page
  echo "<!DOCTYPE html>
<html>
<body>

<h1> Probabilistic hindcasts of ${year} September mean Arctic sea ice extent <br>
based on recalibrated damped anomaly persistence forecast
</h1>

<img src="hindcast_${year}_outlook.png" width=1000 >
<br>

<a href="https://www.elic.ucl.ac.be/users/fmasson/sea-ice-forecasts/oper.html">Back to operational</a>

</body>
</html>
  " > webpages/${year}.html
done

# Finish hindast page

echo "</body>
</html>
" >> webpages/hindcasts.html


rsync -rlptoDv webpages/* $outdir/
# Sync doc
rsync -rlptoDv ./doc/APPLICATE-benchmark.pdf $outdir/
echo "
Visit >>>>  www.climate.be/users/fmasson/sea-ice-forecasts/oper.html   <<<<
"

