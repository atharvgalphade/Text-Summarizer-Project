import os
from TextSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset,load_from_disk
from TextSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config=DataTransformationConfig):
        # We are initializing this helper with some settings (config).
        # These settings tell us where the data is, and which tokenizer to use.
        self.config = config
        
        # We are using a special "tokenizer" to break text into smaller pieces (tokens).
        # It's like cutting big sentences into words or tiny understandable bits.
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        
    def convert_examples_to_features(self, example_batch):
    # Here, we take one small group (batch) of data, with dialogues and summaries.

    # We break the dialogues into tokens (pieces of text) so the machine can understand it.
    # max_length=1024 means we don’t allow dialogues to be too long.
        input_encodings = self.tokenizer(
        example_batch['dialogue'], 
        max_length=1024, 
        truncation=True
    )
    
    # Now we break the summaries into tokens too. Summaries are shorter, so max_length=128.
    # We tell the tokenizer, "You are now cutting text for answers (targets)."
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'], 
                max_length=128, 
                truncation=True
            )
        
        # We return the "ingredients" needed for the model: input_ids, attention_masks, and labels.
        # - input_ids: Numbers that represent the words in the dialogue.
        # - attention_mask: Helps the machine know which words to focus on.
        # - labels: Numbers that represent the words in the summary.
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    
    
    def convert(self):
        # Load the entire dataset from the disk.
        # It’s like opening a box of data that has all the dialogues and summaries.
        dataset_samsum = load_from_disk(self.config.data_path)
        
        # Use the function above (convert_examples_to_features) to process each group of data.
        # 'map' means we apply the function to all the batches in the dataset.
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        
        # Save the processed data back to disk for later use.
        # We create a new folder inside the root directory called "samsum_dataset".
        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum_dataset")
        )
