#!/usr/bin/env bash
set -e
source /temp/miniconda/bin/activate
conda activate LENSR

set -x

###### Download & Preprocess glove ######
echo ###### Download & Preprocess glove ######
if [ -d ~/LENSR_data ]; then
	if [ -f ~/LENSR_data/glove.6B.50d.txt ]; then
		echo -n
	else
		cd ~/LENSR_data
		wget http://nlp.stanford.edu/data/glove.6B.zip
		unzip glove.6B.zip glove.6B.50d.txt
		cd -
	fi
	if [ -d ~/LENSR_data/glove.6B.50d.dat ]; then
		echo -n
	else
		cp glove.py ~/LENSR_data
		cd ~/LENSR_data
		python glove.py
		cd -
	fi
	mkdir -p ../dataset/glove
	if [ -f ../dataset/glove/glove.6B.50d.dat ]; then
		echo -n
	else
		cp ~/LENSR_data/glove.6B.50d.idx.pkl ../dataset/glove/
		cp ~/LENSR_data/glove.6B.50d.words.pkl ../dataset/glove/
		cp -r ~/LENSR_data/glove.6B.50d.dat ../dataset/glove/glove.5B.50d.dat
	fi
else
	wget http://nlp.stanford.edu/data/glove.6B.zip
	unzip glove.6B.zip
	python glove.py
	mv glove.6B.50d.idx.pkl ../dataset/glove/
	mv glove.6B.50d.words.pkl ../dataset/glove/
	mv glove.6B.50d.dat ../dataset/glove/glove.6B.50d.dat
	rm glove.6B.zip
	rm *.txt
fi

###### Download VRD dataset ######
if [ -d ~/LENSR_data ]; then
	if [ -d ~/LENSR_data/sg_dataset/sg_train_images ]; then
		echo -n
	else
		cd ~/LENSR_data
		wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
		unzip sg_dataset.zip
		cd -
	fi
	mkdir -p ../dataset/VRD/sg_dataset/
	if [ -d ../dataset/VRD/sg_dataset/sg_train_images ]; then
		echo -n
	else
		cp -r ~/LENSR_data/sg_dataset/sg_train_images ../dataset/VRD/sg_dataset/sg_train_images
		cp -r ~/LENSR_data/sg_dataset/sg_test_images ../dataset/VRD/sg_dataset/sg_test_images
	fi

	cd ../tools
	python preprocess_image.py
	python remove_empty_sample.py
	cd -
else
	wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
	unzip sg_dataset.zip
	mv sg_dataset/sg_train_images ../dataset/VRD/sg_dataset/sg_train_images
	mv sg_dataset/sg_test_images ../dataset/VRD/sg_dataset/sg_test_images

	cd ../tools
	python preprocess_image.py
	python remove_empty_sample.py
	cd -
	rm -rf sg_dataset
	rm sg_dataset.zip
fi

###### Create neccessary data ######
cd ../tools
python find_rels.py
python tokenize_vocabs.py
python rel2cnf.py
python cnf2ddnnf.py --ds_name vrd --save_path ../dataset/VRD/
python relcnf2data.py
python relddnnf2data.py
cd -

###### Train embedder ######
cd ../model/pygcn/pygcn

atom_options=("" "_ddnnf")
dataset_options=("vrd")

ind_options="--indep_weight"
reg_options="--w_reg 0.1"
non_reg_options="--w_reg 0.0"

for dataset in ${dataset_options[@]}; do
    for atom in "${atom_options[@]}"; do
        if [[ "${atom}" == "" ]]; then
            python train.py --ds_path ../../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${non_reg_options} ${ind_options}
            python train.py --ds_path ../../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${non_reg_options}
        fi
        if [[ ${atom} == '_ddnnf' ]]; then
            python train.py --ds_path ../../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${reg_options} ${ind_options} 
            python train.py --ds_path ../../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${reg_options}
            python train.py --ds_path ../../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${ind_options}
            python train.py --ds_path ../../../dataset/VRD --dataset ${dataset}${atom} --epochs 15 --dataloader_worker 5 ${non_reg_options}
        fi
    done
done
cd -

###### Train relation predictor ######
cd ../model/relation_prediction

atom_options=("" "_ddnnf")
dataset_options=("vrd")

ind_options=".ind"
reg_options=".reg0.1"
non_reg_options=".reg0.0"

for dataset in ${dataset_options[@]}; do
    for atom in "${atom_options[@]}"; do
        if [[ "${atom}" == "" ]]; then
            python train.py --epochs 100 --filename "${dataset}${atom}${non_reg_options}.model" --ds_name vrd
            python train.py --epochs 100 --filename "${dataset}${atom}${non_reg_options}${ind_options}.model" --ds_name vrd
        fi
        if [[ ${atom} == '_ddnnf' ]]; then
            python train.py --epochs 100 --filename "${dataset}${atom}${non_reg_options}.model" --ds_name vrd_ddnnf
            python train.py --epochs 100 --filename "${dataset}${atom}${non_reg_options}${ind_options}.model" --ds_name vrd_ddnnf
            python train.py --epochs 100 --filename "${dataset}${atom}${reg_options}.model" --ds_name vrd_ddnnf
            python train.py --epochs 100 --filename "${dataset}${atom}${reg_options}${ind_options}.model" --ds_name vrd_ddnnf
        fi
    done
done
cd -
