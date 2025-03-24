# Puzzle CAPTCHA Solver

## Problema
I CAPTCHA di nuova generazione sono più sicuri dei CAPTCHA tradizionali?

## Tesi
È possibile implementare dei bot in grado di automatizzare la loro risoluzione.

## Motivazione e rilevanza
I CAPTCHA sono sistemi di sicurezza progettati per distinguere tra utenti umani e bot. Questi vengono utilizzati per prevenire attività automatizzate indesiderate, come spam, attacchi DDoS e malvertising.

Con l'avanzamento dell'intelligenza artificiale, i CAPTCHA tradizionali sono diventati vulnerabili agli attacchi automatizzati. Questo progetto dimostra la possibilità di bypassare i CAPTCHA moderni, evidenziando le loro debolezze e contribuendo allo sviluppo di soluzioni più sicure e intuitive per l'utente finale.

## Struttura del progetto
```
/data:
  /processed → Contiene le immagini processate dagli script
mouse_movement_model.zip → Modello allenato per il movimento del mouse
multi_cls.onnx → Modello YOLO allenato per l’analisi delle immagini
/src:
  /scripts:
    mouse_mover.py → Script con la classe per i movimenti del mouse
    template_matching.py → Script per la risoluzione del CAPTCHA con template matching
    yolo.py → Script per la risoluzione del CAPTCHA con YOLO
/notebooks:
  mouseRL.ipynb → Notebook per la creazione del modello mouse_movement_model.zip
/tests:
  demo.py → Script che testa tutte le combinazioni per la risoluzione del CAPTCHA
/docs:
  report.docx → Report del progetto
```

## Istruzioni per l'uso
### Installazione delle dipendenze
Eseguire il seguente comando per installare le dipendenze richieste:
```bash
pip install -r requirements.txt
```

### Esecuzione degli script
- **Test automatico di tutte le soluzioni**:
  ```bash
  python tests/demo.py
  ```
- **Test delle soluzioni singolarmente**:
  - Template Matching:
    ```bash
    python src/scripts/template_matching.py
    ```
  - YOLO:
    ```bash
    python src/scripts/yolo.py
    ```
  **Nota**: Per eseguire singolarmente template_matching.py o yolo.py, modificare la riga negli import da:
  ```python
  from scripts.mouse_mover import MouseMover
  ```
  a:
  ```python
  from mouse_mover import MouseMover
  ```

### Addestramento del modello per il movimento del mouse
Per addestrare il modello da zero, importare ed eseguire il notebook `mouseRL.ipynb` su Google Colab o un ambiente simile.
