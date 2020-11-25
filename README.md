# multitask-bart

based on https://github.com/patil-suraj/exploring-T5/blob/master/t5_fine_tuning.ipynb

    DATA_DIR=.
    OUTPUT_DIR=bart_res

    python train.py \
        --data_dir $DATA_DIR \
        --output_dir $OUTPUT_DIR \
        --max_seq_length 128 \
        --warmup_steps 500 \
        --train_batch_size 16 \
        --eval_batch_size 16 \
        --num_train_epochs 10 \
        --gradient_accumulation_steps 4 \
        --tasks response,emotion,sentiment \
        --task_dirs dd,tec,sst-2

model

https://huggingface.co/transformers/v2.9.1/model_doc/bart.html

dataset

http://yanran.li/dailydialog

http://saifmohammad.com/WebPages/SentimentEmotionLabeledData.html

https://nlp.stanford.edu/sentiment/index.html

hyperparameter

https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md
