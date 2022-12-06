#!bin/bash
blender=/Applications/Blender.app/Contents/MacOS/Blender

script1=blender/plot_trajectory.py
script2=blender/plot_trajectory_ortho.py

t1="unstable_orbit"
t2="lensing"

uorbCSV="codes/$t1.csv"
lensCSV="codes/$t2.csv"

outDir="SciPost_resubmission/fig/"
# outDir="."

opts="-scale 1.0"

echo "plot $t1 orbit..."
$blender --background --python $script1  -- -save "$outDir/$t1.png" -zscale 0.28 -csvfile $uorbCSV $opts
$blender --background --python $script2 -- -save "$outDir/"$t1"_ortho_1.png" -zscale 0.01 -csvfile $uorbCSV -zview 9 $opts
$blender --background --python $script2 -- -save "$outDir/"$t1"_ortho_2.png" -zscale 0.01 -csvfile $uorbCSV -zview -9 $opts
echo "done."

echo "plot $t2..."
$blender --background --python $script1 -- -save "$outDir/"$t2".png" -zscale 8 -csvfile $lensCSV $opts
$blender --background --python $script2 -- -save "$outDir/"$t2"_ortho_1.png" -zscale 0.01 -csvfile $lensCSV -zview 9 $opts
$blender --background --python $script2 -- -save "$outDir/"$t2"_ortho_2.png" -zscale 0.01 -csvfile $lensCSV -zview -9 $opts
echo "done."