#!/bin/bash

subjectlist=hcp1200.txt

while read -r subject;
do
    mkdir -p hcp/$subject
    mkdir -p hcp/$subject/T1w

    aws s3 cp s3://hcp-openaccess/HCP_1200/$subject/T1w/T1w_acpc_dc_restore_brain.nii.gz hcp/$subject/T1w 
    aws s3 cp s3://hcp-openaccess/HCP_1200/$subject/T1w/T2w_acpc_dc_restore_brain.nii.gz hcp/$subject/T1w 

done < $subjectlist
