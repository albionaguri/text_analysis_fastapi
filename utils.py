import argparse
import os

# INPUT
# permettere all’utente di scegliere tra due modalità (usare il modulo argparse):
# 1. tramite riga di comando (come argomento)
# 2. tramite un file che contiene il testo, specificandone il filepath


def validate_input_file(filepath):
    # validazione del file di input
    if not os.path.exists(filepath):
        raise argparse.ArgumentTypeError(f"Il file {filepath} non esiste")
    
    if not filepath.endswith(".txt"):
        raise argparse.ArgumentTypeError(f"Il file {filepath} non ha un'estensione supportata  (.txt)")

    return filepath


def load_text(filepath):
    # Carica il testo dal file specificato
    if not os.path.exists(filepath):
        raise ValueError(f"Il file {filepath} non esiste")
    
    if not filepath.endswith(".txt"):
        raise ValueError(f"Il file {filepath} non ha un'estensione supportata  (.txt)")

    with open(filepath, "r") as file:
        text = file.read().strip()
    if not text:
        raise ValueError(f"Il file {filepath} è vuoto")
    
    return text

def save_output(results, output_filepath):
    # salve i risultati dell'analisi in un file
    with open(output_filepath, 'w') as file:
        for result in results:
            file.write(f"{result}\n")
