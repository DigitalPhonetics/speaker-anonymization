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

vp_dir=../Voice-Privacy-Challenge-2020/baseline
cd ${vp_dir}

. utils/parse_options.sh || exit 1;

. path.sh
. cmd.sh

corpus_dir=corpora
data_dir=data

#=========== end config ===========

mkdir -p ${data_dir}
cp utils/parse_options.sh .

# Make evaluation data
printf "${GREEN}\nMaking evaluation subsets...${NC}\n"
temp=$(mktemp)
for suff in dev test; do
  for name in ${corpus_dir}/libri_$suff/{enrolls,trials_f,trials_m} \
      ${corpus_dir}/vctk_$suff/{enrolls_mic2,trials_f_common_mic2,trials_f_mic2,trials_m_common_mic2,trials_m_mic2}; do
    [ ! -f $name ] && echo "File $name does not exist" && exit 1
  done

  dset_in=${corpus_dir}/libri_$suff
  dset_out=${data_dir}/libri_$suff
  utils/subset_data_dir.sh --utt-list ${dset_in}/enrolls ${dset_in} ${dset_out}_enrolls || exit 1
  cp ${dset_in}/enrolls ${dset_out}_enrolls || exit 1

  cut -d' ' -f2 ${dset_in}/trials_f | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ${dset_in} ${dset_out}_trials_f || exit 1
  cp ${dset_in}/trials_f ${dset_out}_trials_f/trials || exit 1

  cut -d' ' -f2 ${dset_in}/trials_m | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ${dset_in} ${dset_out}_trials_m || exit 1
  cp ${dset_in}/trials_m ${dset_out}_trials_m/trials || exit 1

  utils/combine_data.sh ${dset_out}_trials_all ${dset_out}_trials_f ${dset_out}_trials_m || exit 1
  cat ${dset_out}_trials_f/trials ${dset_out}_trials_m/trials > ${dset_out}_trials_all/trials

  dset_in=${corpus_dir}/vctk_$suff
  dset_out=${data_dir}/vctk_$suff
  utils/subset_data_dir.sh --utt-list ${dset_in}/enrolls_mic2 ${dset_in} ${dset_out}_enrolls || exit 1
  cp ${dset_in}/enrolls_mic2 ${dset_out}_enrolls/enrolls || exit 1

  cut -d' ' -f2 ${dset_in}/trials_f_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ${dset_in} ${dset_out}_trials_f || exit 1
  cp ${dset_in}/trials_f_mic2 ${dset_out}_trials_f/trials || exit 1

  cut -d' ' -f2 ${dset_in}/trials_f_common_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ${dset_in} ${dset_out}_trials_f_common || exit 1
  cp ${dset_in}/trials_f_common_mic2 ${dset_out}_trials_f_common/trials || exit 1

  utils/combine_data.sh ${dset_out}_trials_f_all ${dset_out}_trials_f ${dset_out}_trials_f_common || exit 1
  cat ${dset_out}_trials_f/trials ${dset_out}_trials_f_common/trials > ${dset_out}_trials_f_all/trials

  cut -d' ' -f2 ${dset_in}/trials_m_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ${dset_in} ${dset_out}_trials_m || exit 1
  cp ${dset_in}/trials_m_mic2 ${dset_out}_trials_m/trials || exit 1

  cut -d' ' -f2 ${dset_in}/trials_m_common_mic2 | sort | uniq > $temp
  utils/subset_data_dir.sh --utt-list $temp ${dset_in} ${dset_out}_trials_m_common || exit 1
  cp ${dset_in}/trials_m_common_mic2 ${dset_out}_trials_m_common/trials || exit 1

  utils/combine_data.sh ${dset_out}_trials_m_all ${dset_out}_trials_m ${dset_out}_trials_m_common || exit 1
  cat ${dset_out}_trials_m/trials ${dset_out}_trials_m_common/trials > ${dset_out}_trials_m_all/trials

  utils/combine_data.sh ${dset_out}_trials_all ${dset_out}_trials_f_all ${dset_out}_trials_m_all || exit 1
  cat ${dset_out}_trials_f_all/trials ${dset_out}_trials_m_all/trials > ${dset_out}_trials_all/trials
done
rm $temp
rm parse_options.sh
