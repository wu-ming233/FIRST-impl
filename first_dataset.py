from torchtune.datasets import InstructDataset
from torchtune.data import InstructTemplate, Message

class FIRSTInstructTemplate(InstructTemplate):
    template = "Rank the passages below based on their relevance to the search query: {query}\n\n{passages}\nSearch Query: {query}\nRank the {num_passages} passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. The output format should be [] > [], e.g., [2] > [1], Only respond with the ranking results, do not say any word or explain."

    @classmethod
    def format(cls, sample, column_map=None):
        query = sample['query']
        passages = '\n\n'.join([f"[{i+1}] {p}" for i, p in enumerate(sample['passages'])])
        num_passages = len(sample['passages'])
        return cls.template.format(query=query, passages=passages, num_passages=num_passages)

class FIRSTDataset(InstructDataset):
    def __init__(self, tokenizer, max_seq_len=512, split="train"):
        super().__init__(
            tokenizer=tokenizer,
            source="castorini/rank_zephyr_training_data",
            template=FIRSTInstructTemplate,
            max_seq_len=max_seq_len,
            split=split
        )
    
    def __getitem__(self, idx):
        item = self._data[idx]
        conversations = item['conversations']
        
        human_message = next(msg for msg in conversations if msg['from'] == 'human')
        query_and_passages = human_message['value'].split('\n\n')
        query = query_and_passages[0].split(': ')[1]
        passages = query_and_passages[1:-1]
        
        gpt_message = next(msg for msg in conversations if msg['from'] == 'gpt')
        rankings = gpt_message['value'].split(' > ')
        rankings = [int(rank.strip('[]')) for rank in rankings]
        
        sample = {
            'query': query,
            'passages': passages,
            'output': ' > '.join([f'[{r}]' for r in rankings])
        }
        
        prompt = self.template.format(sample)
        
        message = Message(role="user", content=prompt)
        
        tokens, attention_mask = self._tokenizer.tokenize_messages([message], max_seq_len=self.max_seq_len)
        
        return {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "rankings": rankings
        }

def first_dataset(tokenizer, max_seq_len=512, split="train"):
    return FIRSTDataset(tokenizer=tokenizer, max_seq_len=max_seq_len, split=split)