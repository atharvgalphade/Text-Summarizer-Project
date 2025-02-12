from TextSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config=ConfigurationManager().get_model_evaluation_config()
        
    def predict(self,text):
        tokenizer=AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {
            "length_penalty": 0.2,  # Controls the length of the summary; lower values encourage shorter summaries
            "num_beams": 8,         # Number of beams for beam search; higher value improves quality but is slower
            "max_length": 128       # Maximum number of tokens in the summary
        }
        pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)

        print("Dialogue:")
        print(text)
        
        output=pipe(text,**gen_kwargs)[0]["summary_text"]
        print("\nModel Summary:")
        print(output)
        
        return output