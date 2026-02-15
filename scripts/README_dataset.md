# Generisanje dataseta za phishing detekciju

## Format (isti kao `raw.csv`)

| Kolona   | Opis                                      |
|----------|-------------------------------------------|
| ender    | Pošiljalac (npr. `Ime <email@domen.com>`) |
| receiver | Primaoc (email)                            |
| date     | Datum (RFC format)                         |
| subject  | Predmet                                   |
| body     | Tijelo maila                              |
| label    | `0` = legitimni, `1` = phishing           |
| urls     | Broj URL-ova u mailu (ceo broj)           |

## Pokretanje

```bash
python scripts/generate_dataset.py
```

Izlaz: `data/raw_generated.csv` (oko 26.000 redova: 14.000 legitimnih, 12.000 phishing).

## Korištenje za treniranje

- **Samo novi dataset:** preimenuj `raw_generated.csv` u `raw.csv` (ili u `config.py` postavi `RAW_DATA_PATH` na `data/raw_generated.csv`) pa pokreni treniranje.
- **Spoj sa starim:** u Pythonu učitaj oba CSV-a, spoji (`pd.concat`), izmiješaj redove, snimi kao novi `raw.csv` pa treniraj.

## Dizajn dataseta (prema sugestijama profesora)

- **Legitimni mailovi (label=0):** radni mailovi, newsletteri, obavijesti, bez fraza tipa "verify account" / "urgent" / "click here now", sa normalnim domenima (company.com, gmail.com, itd.).
- **Phishing mailovi (label=1):** tipični signali (hitnost, lažni PayPal/bank/Apple, verify/login, sumnjivi linkovi, domeni .tk, .xyz, .pw, itd.).

Cilj je da model nauči da **ne** označava obične mailove kao phishing, a da i dalje dobro detektuje pravi phishing.
