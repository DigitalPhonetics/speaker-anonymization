#!/bin/bash
# Extract of Voice-Privacy-Challenge-2020/baseline/run.sh
#
# License of the original script:
# Copyright (C) 2020  <Brij Mohan Lal Srivastava, Natalia Tomashenko, Xin Wang, Jose Patino, Paul-Gauthier Noé, Andreas Nautsch, ...>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#


set -e

#===== begin config =======

nj=$(nproc)
split='dev'

vp_dir=Voice-Privacy-Challenge-2020/baseline
cd $vp_dir

. utils/parse_options.sh || exit 1;

. path.sh
. cmd.sh

anon_data_suffix=_anon

#=========== end config ===========

# Make VCTK anonymized evaluation subsets
printf "${GREEN}\nMaking VCTK anonymized evaluation subsets for ${split}...${NC}\n"
temp=$(mktemp)

dset=data/vctk_$split
for name in ${dset}_trials_f_all$anon_data_suffix ${dset}_trials_m_all$anon_data_suffix; do
  [ ! -d $name ] && echo "Directory $name does not exist" && exit 1
done

cut -d' ' -f2 ${dset}_trials_f/trials | sort | uniq > $temp
utils/subset_data_dir.sh --utt-list $temp ${dset}_trials_f_all$anon_data_suffix ${dset}_trials_f${anon_data_suffix} || exit 1
cp ${dset}_trials_f/trials ${dset}_trials_f${anon_data_suffix} || exit 1

cut -d' ' -f2 ${dset}_trials_f_common/trials | sort | uniq > $temp
utils/subset_data_dir.sh --utt-list $temp ${dset}_trials_f_all$anon_data_suffix ${dset}_trials_f_common${anon_data_suffix} || exit 1
cp ${dset}_trials_f_common/trials ${dset}_trials_f_common${anon_data_suffix} || exit 1

cut -d' ' -f2 ${dset}_trials_m/trials | sort | uniq > $temp
utils/subset_data_dir.sh --utt-list $temp ${dset}_trials_m_all$anon_data_suffix ${dset}_trials_m${anon_data_suffix} || exit 1
cp ${dset}_trials_m/trials ${dset}_trials_m${anon_data_suffix} || exit 1

cut -d' ' -f2 ${dset}_trials_m_common/trials | sort | uniq > $temp
utils/subset_data_dir.sh --utt-list $temp ${dset}_trials_m_all$anon_data_suffix ${dset}_trials_m_common${anon_data_suffix} || exit 1
cp ${dset}_trials_m_common/trials ${dset}_trials_m_common${anon_data_suffix} || exit 1

rm $temp