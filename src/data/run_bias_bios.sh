
# Encode text using Bert

for split in 'train' 'dev' 'test'
do
    python3 encode_bert_states.py \
    --input_file ../../data/biasbios/EN/$split.pickle \
    --output_dir ../../data/bert_encode_biasbios/EN/ \
    --split $split
done
