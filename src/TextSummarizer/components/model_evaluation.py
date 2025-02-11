from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset,load_from_disk
import evaluate # Used in place of load_metric from datasets. works in similar manner but load_metric is not available in latest version of datasets.
import torch
import pandas as pd 
from tqdm import tqdm 
from TextSummarizer.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self,config:ModelEvaluationConfig):
        self.config=config
    
    def generate_batch_sized_chunks(self,list_of_elements, batch_size):
        """
        Splits a list into smaller chunks of the specified batch size.

        Args:
            list_of_elements (list): The list to split into chunks.
            batch_size (int): The size of each chunk.

        Yields:
            list: A chunk of the list with length equal to batch_size.
        """
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i:i+batch_size]

    # Function to evaluate the model on the test dataset and compute a given metric
    def calculate_metric_on_test_ds(self,dataset, metric, model, tokenizer,
                                    batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu",
                                    column_text='article',
                                    column_summary='highlights'):
        """
        Evaluates a summarization model on a test dataset and calculates the given metric.

        Args:
            dataset (Dataset): The dataset containing articles and reference summaries.
            metric (Metric): The metric to compute (e.g., ROUGE, BLEU).
            model (PreTrainedModel): The trained summarization model.
            tokenizer (PreTrainedTokenizer): The tokenizer for encoding and decoding text.
            batch_size (int): The number of samples to process at once. Default is 16.
            device (torch.device): The device (CPU/GPU) for model computation.
            column_text (str): The column name in the dataset containing the input text (articles).
            column_summary (str): The column name in the dataset containing reference summaries.

        Returns:
            dict: The computed metric scores.
        """
        # Split the input articles into batches of size `batch_size`
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        # Split the target summaries into batches of size `batch_size`
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        # Loop through each batch of articles and their corresponding summaries
        for article_batch, target_batch in tqdm(
                zip(article_batches, target_batches), total=len(article_batches)):

            # Tokenize the batch of articles
            inputs = tokenizer(article_batch,
                            max_length=1024,  # Maximum length of input sequences
                            truncation=True,  # Truncate input sequences longer than max_length
                            padding='max_length',  # Pad input sequences to max_length
                            return_tensors='pt')  # Return PyTorch tensors

            # Generate summaries using the model
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),  # Move input IDs to the specified device
                attention_mask=inputs["attention_mask"].to(device),  # Move attention mask to device
                length_penalty=0.8,  # Encourage shorter summaries (lower length penalty)
                num_beams=8,  # Use beam search with 8 beams for better summaries
                max_length=128  # Maximum length of generated summaries
            )

            # Decode the generated summaries into text
            decoded_summaries = [
                tokenizer.decode(s,
                                skip_special_tokens=True,  # Remove special tokens (e.g., <s>, </s>)
                                clean_up_tokenization_spaces=True)  # Clean up spaces in decoded text
                for s in summaries
            ]

            # Replace empty strings in decoded summaries with a single space (clean-up step)
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]

            # Add the predictions and reference summaries to the metric for evaluation
            metric.add_batch(predictions=decoded_summaries, references=target_batch)

        # Compute the metric scores (e.g., ROUGE scores)
        score = metric.compute()
        return score
    
    def evaluate(self):
        device="cuda" if torch.cuda.is_available() else "cpu"
        tokenizer=AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model_pegasus=AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
        
        dataset_samsum_pt=load_from_disk(self.config.data_path)
        batch_size=16
        rouge_names=["rouge1","rouge2","rougeL","rougeLsum"]
        rouge_metric=evaluate.load('rouge')
        
        score=self.calculate_metric_on_test_ds(
            dataset_samsum_pt['test'][0:10], rouge_metric, model_pegasus, tokenizer, batch_size, column_text='dialogue', column_summary='summary'
        )

        rouge_dict=dict((rn, score[rn]) for rn in rouge_names)

        df=pd.DataFrame(rouge_dict, index=[f'pegasus'])
        df.to_csv(self.config.metric_file_name,index=False)