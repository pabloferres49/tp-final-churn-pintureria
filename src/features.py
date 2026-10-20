"""
Construcción del dataset de modelado: un panel cliente-mes.

Cada fila es una "foto" del cliente al cierre del mes t, con atributos calculados
SOLO con información disponible hasta ese cierre (sin fuga de datos), y la etiqueta
`churn_mes_siguiente` = 1 si el cliente NO realiza ninguna compra en el mes t+1.
"""
import numpy as np
import pandas as pd

TIPOS_COMPRA = ["FC", "TK"]          # facturas y tickets = compras
TIPO_DEVOLUCION = "NC"               # notas de crédito = devoluciones/ajustes
MARCA_PROPIA = "Sinteplast"


def cargar_ventas(ruta: str) -> pd.DataFrame:
    """Lee el CSV anonimizado y agrega columnas auxiliares."""
    df = pd.read_csv(ruta, parse_dates=["FECHA"], dtype={"COD ARTICULO": str, "ID_CLIENTE": str})
    df["MES"] = df["FECHA"].dt.to_period("M")
    df["ID_COMPROBANTE"] = (df["PUNTO DE VENTA"].astype(str) + "-" + df["TIPO"] + "-"
                            + df["LETRA"] + "-" + df["NUMERO COMPROBANTE"].astype(str))
    df["COSTO_TOTAL"] = df["COSTO"] * df["CANTIDAD"]
    df["ES_COMPRA"] = df["TIPO"].isin(TIPOS_COMPRA)
    df["ES_SINTEPLAST"] = (df["MARCA"] == MARCA_PROPIA).astype(int)
    return df


def construir_panel(df: pd.DataFrame, excluir=("CONSUMIDOR_FINAL",)) -> pd.DataFrame:
    """Devuelve el panel cliente-mes con features y target."""
    df = df[~df["ID_CLIENTE"].isin(excluir)].copy()
    meses = pd.period_range(df["MES"].min(), df["MES"].max(), freq="M")

    compras = df[df["ES_COMPRA"]]
    # Ventas totales del local por mes (todos los clientes, incluido consumidor final se
    # excluyó arriba -> usamos el df de clientes identificados como referencia de mercado)
    ventas_local_mes = compras.groupby("MES")["PRECIO TOTAL"].sum()
    tickets_local_mes = compras.groupby("MES")["ID_COMPROBANTE"].nunique()

    primera_compra = compras.groupby("ID_CLIENTE")["FECHA"].min()
    filas = []

    for t in meses[:-1]:                     # el último mes no tiene t+1 observable
        fin_t = t.to_timestamp(how="end").normalize()
        ventana_3m = pd.period_range(t - 2, t, freq="M")

        hist = df[df["MES"] <= t]
        hist_c = hist[hist["ES_COMPRA"]]
        clientes = primera_compra[primera_compra <= fin_t].index
        if len(clientes) == 0:
            continue

        h3 = hist_c[hist_c["MES"].isin(ventana_3m)]
        h1 = hist_c[hist_c["MES"] == t]
        local_3m = ventas_local_mes.reindex(ventana_3m).fillna(0).sum()
        tk_local_3m = tickets_local_mes.reindex(ventana_3m).fillna(0).sum()
        ticket_local_3m = local_3m / tk_local_3m if tk_local_3m else np.nan

        g_hist = hist_c.groupby("ID_CLIENTE")
        g3 = h3.groupby("ID_CLIENTE")

        f = pd.DataFrame(index=pd.Index(clientes, name="ID_CLIENTE"))
        f["recencia_dias"] = (fin_t - g_hist["FECHA"].max()).dt.days
        f["antiguedad_dias"] = (fin_t - g_hist["FECHA"].min()).dt.days
        f["compro_mes_actual"] = f.index.isin(h1["ID_CLIENTE"]).astype(int)
        f["meses_activos_3m"] = g3["MES"].nunique()
        f["comprobantes_3m"] = g3["ID_COMPROBANTE"].nunique()

        meses_hist = g_hist["MES"].nunique()
        antig_meses = ((t - g_hist["MES"].min()).apply(lambda x: x.n) + 1)
        f["tasa_meses_activos_hist"] = meses_hist / antig_meses

        # Montos relativos al total del local -> neutraliza la inflación
        monto_3m = g3["PRECIO TOTAL"].sum()
        f["share_ventas_3m_pm"] = monto_3m / local_3m * 1000          # por mil
        ticket_3m = monto_3m / g3["ID_COMPROBANTE"].nunique()
        f["ticket_rel_3m"] = ticket_3m / ticket_local_3m
        monto_t = h1.groupby("ID_CLIENTE")["PRECIO TOTAL"].sum()
        f["tendencia_monto"] = (monto_t / (monto_3m / 3)).reindex(f.index)

        # Días promedio entre compras (fechas distintas)
        def _gap(s):
            d = np.sort(s.unique())
            return np.diff(d).mean() / np.timedelta64(1, "D") if len(d) > 1 else np.nan
        f["dias_entre_compras_prom"] = g_hist["FECHA"].apply(_gap)

        f["descuento_prom_3m"] = g3["DESCUENTO (%)"].mean()
        f["articulos_distintos_3m"] = g3["COD ARTICULO"].nunique()
        f["pct_lineas_sinteplast"] = g_hist["ES_SINTEPLAST"].mean()
        f["pct_factura_A"] = g_hist["LETRA"].apply(lambda s: (s == "A").mean())

        bruto = g_hist["PRECIO TOTAL"].sum()
        costo = g_hist["COSTO_TOTAL"].sum()
        f["margen_bruto_pct"] = (bruto - costo) / bruto.replace(0, np.nan)

        dev = hist[hist["TIPO"] == TIPO_DEVOLUCION].groupby("ID_CLIENTE")["PRECIO TOTAL"].sum().abs()
        f["pct_devoluciones"] = (dev / bruto.replace(0, np.nan)).reindex(f.index).fillna(0)

        f["vendedor_principal"] = g_hist["COD VENDEDOR"].agg(lambda s: s.mode().iat[0]).astype(str)

        # Target: ¿compra en t+1?
        compradores_t1 = set(compras.loc[compras["MES"] == t + 1, "ID_CLIENTE"])
        f["churn_mes_siguiente"] = (~f.index.isin(compradores_t1)).astype(int)

        f["mes_foto"] = str(t)
        filas.append(f.reset_index())

    panel = pd.concat(filas, ignore_index=True)
    # NaN por ausencia de compras en la ventana -> 0 tiene sentido de negocio
    for c in ["meses_activos_3m", "comprobantes_3m", "share_ventas_3m_pm",
              "tendencia_monto", "articulos_distintos_3m"]:
        panel[c] = panel[c].fillna(0)
    # ticket_rel_3m, descuento_prom_3m, dias_entre_compras_prom, margen_bruto_pct
    # quedan con NaN y se imputan dentro del Pipeline (mediana) para evitar fuga.
    return panel


FEATURES_NUM = [
    "recencia_dias", "antiguedad_dias", "compro_mes_actual", "meses_activos_3m",
    "comprobantes_3m", "tasa_meses_activos_hist", "share_ventas_3m_pm", "ticket_rel_3m",
    "tendencia_monto", "dias_entre_compras_prom", "descuento_prom_3m",
    "articulos_distintos_3m", "pct_lineas_sinteplast", "pct_factura_A",
    "margen_bruto_pct", "pct_devoluciones",
]
FEATURES_CAT = ["vendedor_principal"]
TARGET = "churn_mes_siguiente"
