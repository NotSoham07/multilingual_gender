
# Encode text using Bert
for split in 'train' 'dev' 'test'
do
    python encode_bert_states.py \
        --input_file ../../data/biasbios/IT/$split.pickle \
        --output_dir ../../data/bert_encode_biasbios/IT\
        --split $split
done
