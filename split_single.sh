#!/bin/bash -e

LD_LIBRARY_PATH=/scratch/bbecker/ffmpeg-build/lib:/usr/local/cuda/lib64/

prefix="/scratch/bbecker"
dest="/data/others/bbecker/frames4"

file="$prefix/video_new/$1"
filen="$1"

mkdir -p $prefix/frames4/$filen
echo "$filen: spliting frames"
/scratch/bbecker/ffmpeg-build/bin/ffmpeg -hide_banner -loglevel panic -stats -y -i $file -qscale:v 2 $prefix/frames4/$filen/${filen}_%06d.jpg

pushd .


echo "$filen: packing frames"
cd $prefix/frames4/$filen/
tar cf $filen.tar .

echo "$filen: moving to data"
mv -- $filen.tar $dest/

popd
rm -rf $prefix/frames4/$filen/

echo "$filen: done"

