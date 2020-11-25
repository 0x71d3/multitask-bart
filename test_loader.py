from torch.utils.data import ConcatDataset, DataLoader
from transformers import BartTokenizer

from dataset import TaskDataset
from sampler import MultitaskBatchSampler

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

tasks = ['response', 'emotion', 'sentiment']
task_dirs = ['dd', 'tec', 'sst-2']

type_path = 'test'
max_len = 128

datasets = []
for task, task_dir in zip(tasks, task_dirs):
    dataset = TaskDataset(task, tokenizer, task_dir, type_path, max_len)
    datasets.append(dataset)

concat_dataset = ConcatDataset(datasets)

batch_size = 16
sampler = MultitaskBatchSampler(concat_dataset, batch_size, True)

loader = DataLoader(concat_dataset, batch_sampler=sampler)

print(len(loader))
# print(*list(loader), sep='\n')
