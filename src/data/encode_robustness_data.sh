
# Encode text using Bert
for split in 'train' 'dev' 'test'
do
    python encode_bert_states.py \
        --input_file ../../data/robustness/data/$split.pickle \
        --output_dir ../../data/robustness/bert_encode_data/ \
        --split $split
done
