import os
import sys

input_dir, output_dir = sys.argv[1:]

for split in ['train', 'validation', 'test']:
    pairs = []
    with open(os.path.join(input_dir, split, f'dialogues_{split}.txt')) as f:
        for line in f:
            texts = line.split('__eou__')[:-1]
            for i in range(len(texts) - 1):
                pairs.append((texts[i].strip(), texts[i+1].strip()))
    
    if split == 'validation':
        split = 'val'

    with open(os.path.join(output_dir, split + '.tsv'), 'w') as f:
        for utterance, response in pairs:
            f.write(utterance + '\t' + response + '\n')
