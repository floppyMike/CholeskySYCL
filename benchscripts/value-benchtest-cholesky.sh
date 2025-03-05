#!/bin/bash

#SBATCH -w simcl1n1,simcl1n2
#SBATCH --job-name="value"
#SBATCH --output=job-value.out
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --exclusive

base="/scratch/blochml"
benchfilessuffix=""

TMP=/scratch/blochml/tmp
TMPDIR=/scratch/blochml/tmp
tempdir=/scratch/blochml/tmp

module load cuda/11.8.0
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

. /scratch/blochml/spack/share/spack/setup-env.sh

if ! spack find 'hipsycl@24.02.0+cuda+ipo' > /dev/null 2>&1; then
	spack install --no-checksum 'hipsycl@=24.02.0+cuda+ipo'
fi

spack load cmake
spack load hipsycl

./create_env.sh CholeskySYCL
. "envs/CholeskySYCL/bin/activate"

cd CholeskySYCL
mkdir -p build

cd /scratch/blochml/CholeskySYCL/build
cmake -DUSE_DOUBLE=ON ..
cmake --build .
cd /scratch-simcl1/blochml/

/scratch/blochml/CholeskySYCL/build/06

cd /scratch/blochml/CholeskySYCL/build
cmake -DUSE_DOUBLE=OFF ..
cmake --build .
cd /scratch-simcl1/blochml/

/scratch/blochml/CholeskySYCL/build/06
