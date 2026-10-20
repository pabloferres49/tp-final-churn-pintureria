"""
Anonimiza el dataset original de ventas antes de publicarlo.

- Elimina nombres de clientes y vendedores.
- Reemplaza el código de cliente por un hash SHA-256 truncado (con sal).
- Corrige problemas de encoding en las descripciones (ej. 'NÂ°' -> 'N°').

Uso:
    python src/anonimizar.py --input data/raw/Data_Set_Pintureria_normalizado.xlsx \
                             --output data/processed/ventas_anonimizado.csv

La sal se lee de la variable de entorno ANON_SALT y NO se publica.
El archivo original (data/raw/) está excluido del repo vía .gitignore.
"""
import argparse
import hashlib
import os

import pandas as pd

HOJA = "DataSet Ventas"
CONSUMIDOR_FINAL = "1"


def _hash(valor: str, sal: str) -> str:
    return "C_" + hashlib.sha256(f"{sal}{valor}".encode()).hexdigest()[:10]


def anonimizar(df: pd.DataFrame, sal: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    cod = df["COD CLIENTE"].astype(str).str.strip()
    df["ID_CLIENTE"] = cod.map(lambda v: "CONSUMIDOR_FINAL" if v == CONSUMIDOR_FINAL else _hash(v, sal))

    df = df.drop(columns=["COD CLIENTE", "NOMBRE CLIENTE", "NOMBRE VENDEDOR"])
    df["DESCRIPCION"] = (df["DESCRIPCION"].astype(str)
                         .str.replace("Â°", "°", regex=False)
                         .str.replace("Âº", "º", regex=False))
    df["HORA"] = df["HORA"].astype(str)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sal = os.environ.get("ANON_SALT", "cambiar-esta-sal")
    df = pd.read_excel(args.input, sheet_name=HOJA)
    out = anonimizar(df, sal)
    out.to_csv(args.output, index=False, encoding="utf-8")
    print(f"OK -> {args.output} ({len(out):,} filas, {out['ID_CLIENTE'].nunique()} clientes)")


if __name__ == "__main__":
    main()
