#!/usr/bin/env bash
set -e
source /temp/miniconda/bin/activate
conda activate LENSR

ds_names=("0303" "0306" "0606")

ulimit -Sn unlimited
echo Set ulimit -Sn unlimited. New max open files is: "$(ulimit -n)"

if [ -z "${SKIP_DATAGEN}" ]; then

echo Generating data

###### General form to CNF, d-DNNF #####
cd ../tools
for ds_name in ${ds_names[@]}; do
    python raw2data.py --save_path ../dataset/Synthetic/ --ds_name "${ds_name}"
done
cd -

###### CNF, d-DNNF to GCN compatible #####
cd ../tools
for ds_name in ${ds_names[@]}; do
    python cnf2data.py --save_path ../dataset/Synthetic/ --ds_name "${ds_name}"
    python ddnnf2data.py --save_path ../dataset/Synthetic/ --ds_name "${ds_name}"
done
cd -

fi

###### Train Synthetic dataset #####
cd ../model/pygcn/pygcn

atom_options=("_0303" "_0306" "_0606" )
dataset_options=("general" "ddnnf" "cnf")

ind_options="--indep_weight"
reg_options="--w_reg 0.1"
non_reg_options="--w_reg 0.0"
all_options="--no-cuda --epochs 15 --dataloader_workers 1 --hidden-layers 5 --ds_path ../../../dataset/Synthetic"

function train_until_good () {
	local filename="log"
	filename="$filename-$(git rev-parse HEAD | head -c 7)"
	filename="$filename-$(date +%c)"
	#filename="$filename-$(printf "%q" "$*" | tr '/' '_')" # These names were too long
	filename="$filename-$BASHPID"
	while ! (python train.py "$@"); do
		echo 'It crashed... retrying'
		sleep 2
	done | tee "$filename"
}

if [ -z "${NUM_PARALLEL}" ]; then
	NUM_PARALLEL=14
fi

function add_train_job () {
	sleep 2
	if [ "${NUM_PARALLEL}" -eq 0 ]; then
		wait -n
	else
		NUM_PARALLEL=$((${NUM_PARALLEL} - 1))
	fi
	train_until_good "$@" &
}

for dataset in ${dataset_options[@]}; do
    for atom in "${atom_options[@]}"; do
        if [[ ${dataset} == 'general' ]]; then
            add_train_job --dataset ${dataset}${atom} ${non_reg_options} ${ind_options} ${all_options}
        fi
        if [[ ${dataset} == 'cnf' ]]; then
            add_train_job --dataset ${dataset}${atom} ${non_reg_options} ${ind_options} ${all_options}
            add_train_job --dataset ${dataset}${atom} ${non_reg_options} ${all_options}
        fi
        if [[ ${dataset} == 'ddnnf' ]]; then
            add_train_job --dataset ${dataset}${atom} ${reg_options} ${ind_options} ${all_options}
            add_train_job --dataset ${dataset}${atom} ${reg_options} ${all_options}
            add_train_job --dataset ${dataset}${atom} ${ind_options} ${all_options}
            add_train_job --dataset ${dataset}${atom} ${non_reg_options} ${all_options}
        fi
    done
done
cd -

wait
