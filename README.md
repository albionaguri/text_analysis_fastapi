# Sentiment Analysis Script

Questo script Python esegue l'analisi del testo utilizzando un modello pre-addestrato di Hugging Face per il riconoscimento delle entità nominate (NER). Può accettare il testo direttamente dalla riga di comando o caricarlo da un file di testo. Offre anche opzioni per filtrare i risultati in base a una soglia di score e per stampare o salvare i risultati.

## Requisiti

- Python 3.12.5
- Linux con venv configurato

## Installazione

1. Clona il repository o scarica il file sentiment_analysis_hf.py.
2. Crea e Attiva di un Ambiente Virtuale:
   python3 -m venv venv
   source venv/bin/activate
3. Installa le dipendenze: pip install -r requirements.txt


## Parametri Opzionali e Obbligatori

Obbligatori:
-t/--text o -f/--file (almeno uno dei due)

Opzionali:
-th/--threshold
-o/--output
--print


## Uttilizo / Esecuzione dello Script di Sentiment Analysis

- Esempi di Utilizzo

1. Usare Testo dalla Riga di Comando:

python sentiment_analysis_hf.py -t "Inserisci il testo da analizzare" --print

2. Usare File di Testo e Salvare i Risultati in un File:

python sentiment_analysis_hf.py -f input.txt -o results.txt

3. Usare Testo con Filtraggio e Stampare i Risultati:

python sentiment_analysis_hf.py -t "Inserire il testo da analizzare" -th 0.9 --print

4. Usare File di Testo con Filtraggio e Salvare i Risultati:

python sentiment_analysis_hf.py -f input.txt -th 0.9 -o filtered_results.txt


## Uttilizo / Esecuzione dello Script di Sentiment Classification

1. Utilizzare Testo dalla Riga di Comando e Stampare i Risultati:

python sentiment_classification_hf.py -t "Testo da inserire per la classificazione dell sentimento" --print

2. Utilizzare Testo da un File e Salvare i Risultati in un File:

python sentiment_classification_hf.py -f input.txt -o results.txt

3. Utilizzare Testo dalla Riga di Comando e Salvare i Risultati in un File:

python sentiment_classification_hf.py -t "Testo da inserire" -o output.txt

4. Utilizzare Testo da un File e Stampare i Risultati:

python sentiment_classification_hf.py -f input.txt --print


## Uttilizo / Esecuzione dello Script di Summarization

1. Eseguire una Sintesi del Testo Fornito tramite Riga di Comando:

python summarization_hf.py -t "Inserire testo da riassumere." --print

2. Eseguire una Sintesi del Testo Presente in un File:

python summarization_hf.py -f long_article.txt --print

3. Salvare la Sintesi del Testo in un File di Output:

python summarization_hf.py -t "Inserire testo da riassumere." -o summary.txt

4. Sintesi di un File di Testo e Salvataggio del Risultato:

python summarization_hf.py -f report.txt -o summarized_report.txt

# OVERVIEW
### Named Entity Recognition (NER) with FastAPI
- L'API consente agli utenti di eseguire NER su dati di testo utilizzando un modello di trasformatore pre-addestrato da Hugging Face.

- Il modello identifica entità quali nomi, organizzazioni, posizioni, ecc. nel testo fornito e restituisce i risultati in formato JSON.

## Features

- Accetta l'input di testo direttamente o da un file.
- Consente di filtrare i risultati in base a una soglia di punteggio.
- Facoltativamente, stampa i risultati direttamente nel terminale.


## Installazione

1.  **Clonare il repository:**
   ```bash
   git clone https://github.com/yourusername/ner-fastapi.git
   cd ner-fastapi

2. Crea un ambiente virtuale:
python3 -m venv venv
source venv/bin/activate

3. Installa i pacchetti richiesti:
pip install fastapi uvicorn transformers

## Running the API
uvicorn main:app --reload

- Apri il browser e vai su http://127.0.0.1:8000/docs per accedere all'interfaccia utente di Swagger, dove puoi testare gli endpoint API.

