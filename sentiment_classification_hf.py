from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_text

# INPUT
# permettere all’utente di scegliere tra due modalità (usare il modulo argparse):
# 1. tramite riga di comando (come argomento)
# 2. tramite un file che contiene il testo, specificandone il filepath

# Load the model and tokenizer for Sentiment Analysis
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Sentiment analysis pipeline
nlp = pipeline('sentiment-analysis', tokenizer=tokenizer, model=model)

def sentiment_analysis(request):
    text = request.text if request.text else load_text(request.file)

    results = nlp(text)
    
    if request.threshold is not None:
        results = [r for r in results if r["score"] >= request.threshold]

    if request.print_result:
        return results

    return {"output": results}



# def perform_analysis(text, threshold=None):

#     # l'analisi del testo usando il modello di HF
#     tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
#     model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

#     ## Sentiment analysis pipeline
#     nlp = pipeline('sentiment-analysis', tokenizer=tokenizer, model=model)

#     results = nlp(text)

#     if threshold is not None:
#         results = [r for r in results if r["score"] >= threshold]

#     return results



# def main():

#     parser = argparse.ArgumentParser(description="Sentiment analysis using a pre-trained model from Hugging Face")  
    
#     parser.add_argument("-t",'--text' ,type=str, help="Testo da analizzare.")
#     parser.add_argument('-f', '--file', type=validate_input_file, help="File contenente il testo da analizzare.")
#     parser.add_argument("-th",'--threshold', type=float, help="Soglia di score per i risultati.", default=None)   
#     parser.add_argument("-o",'--output', type=str, help="File di output per i risultati.")   
#     parser.add_argument("--print", action="store_true", help="Stampa i risultati a schermo.")   

#     args = parser.parse_args()

#     if args.text and args.file:
#         parser.error("Puoi specificare solo uno tra --text e --file.")

#     if not (args.text or args.file):
#         parser.error("Devi specificare --text o --file per fornire il testo da analizzare.")

#     # carica il testo dall'input fornito
#     text = args.text if args.text else load_text(args.file)

#     try:
#         results = perform_analysis(text, args.threshold)

#     except Exception as e:
#         print(f"Errore durante l'analisi del testo: {e}")
#         return
    
#     # OUTPUT
#     # Output completo: risultato completo come generato del modello.
#     # Output filtrato:  risultati con un score superiore a una soglia specificata dall'utente.

#     if args.print:
#         for result in results:
#             print(result)

#     elif args.output:
#         save_output(results, args.output)
#         print(f"Risultati salvati in {args.output}")

#     else:
#         parser.error("Devi specificare se stampare o salvare l'output.")    

# if __name__ == "__main__":
#     main()