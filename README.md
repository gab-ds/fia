# GreenTrails - Modulo AI

Questo repository contiene il codice sorgente per il modulo di
intelligenza artificiale del progetto [C03 GreenTrails](https://github.com/GerardoFesta/GreenTrails).

Progetto combinato Ingegneria del Software/Fondamenti di Intelligenza Artificiale,
a.a. 2023/2024, Università degli Studi di Salerno.

## Partecipanti

|     Team Member     |
|:-------------------:|
| Gabriele Di Stefano |
|  Roberta Galluzzo   |


## Installazione

Python 3.13 è richiesto per l'installazione e l'utilizzo.

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Per installare anche i requisiti previsti dal client mock:

```bash
$ pip install -r helpers/requirements.txt
```

## Esecuzione

```bash
$ uvicorn main:app
```

Il server verrà eseguito in locale, sulla porta 8000.

Per eseguire anche il client mock:

```bash
$ cd helpers
$ python3 mock_system.py
```