case "$1" in
        train)
            python miccai_train_on_new_data.py
            ;;
         build)
            python build_dataset_for_miccai.py -f myfolder/
            ;;
         predict)
            python miccai_segment.py -f pid01.FLAIR.nii.gz -t pid01.mprage.to_FLAIR.nii.gz -m mask.nii.gz -o whm.nii.gz
            ;;
         prerpocess)
            python preproc_FLAIR_MPRAGE.py -t pid01.T1.nii.gz -f pid01.FLAIR.nii.gz -m pid01.maks.nii.gz -p pid01
            ;;
        *)
            echo $"Usage: $0 {train|build|predict|preprocess}"
            exit 1
esac
