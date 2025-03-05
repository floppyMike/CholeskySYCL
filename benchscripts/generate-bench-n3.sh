#!/bin/bash

#SBATCH -w simcl1n3
#SBATCH --job-name="float-gen"
#SBATCH --output=jobfloat-gen.out
#SBATCH --time=48:00:00
#SBATCH --nodes=1

base="/scratch/blochml"

TMP=/scratch/blochml/tmp
TMPDIR=/scratch/blochml/tmp
tempdir=/scratch/blochml/tmp

module load gcc/13.2.0

if [ ! -d "$base" ]; then
	cd /scratch
	git clone --recurse-submodules git@github.tik.uni-stuttgart.de:st177433/BachelorArbeit.git blochml
fi

cd /scratch/blochml

if [ -d "/scratch/blochml/tmp" ]; then
	rm -rf "/scratch/blochml/tmp/*"
fi

branch="row"
git fetch origin
git reset --hard origin/$branch
git switch $branch

./create_env.sh CholeskySYCL
. envs/CholeskySYCL/bin/activate

cd /scratch-simcl1/blochml

echo "Starting with generation..."
python -u /scratch/blochml/CholeskySYCL/py/datagen.py 60000 5 float float-60000
