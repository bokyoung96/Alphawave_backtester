import os
import numpy as np
import pandas as pd

from enum import Enum, unique
from typing import Type, Callable


"""
<NOTE>
1. Refactorizing required by minimizing datas to download through using constituent classficiation datas.
2. Reclassification required for numerous items, considering its characteristics.
3. DB connection / auto-download / auto-update tools required for further code usage.
4. File type optimization required for faster code usage.
"""


@unique
class Market(Enum):
    KOREA = "KOREA"
    KOSPI = "KOSPI"
    KOSPI200 = "KOSPI200"
    KOSDAQ = "KOSDAQ"


@unique
class Asset(Enum):
    stock = "stock"
    bond = "bond"


@unique
class Frequency(Enum):
    n = "n"
    d = "1d"
    m = "1m"
    y = "1y"


@unique
class Source(Enum):
    dataguide = "dataguide"
    quantiwise = "quantiwise"
    bloomberg = "bloomberg"


@unique
class DataPrice(Enum):
    price_o = "price_o"
    price_h = "price_h"
    price_l = "price_l"
    price_c = "price_c"

    float_mktcap = "float_mktcap"

    def as_price(self,
                 exchange: str,
                 asset: str,
                 frequency: str) -> str:
        return f"{exchange.value}_{asset.value}_{self.value}_{frequency.value}"


@unique
class DataVolume(Enum):
    volume = "volume"

    def as_volume(self,
                  exchange: str,
                  asset: str,
                  frequency: str) -> str:
        return f"{exchange.value}_{asset.value}_{self.value}_{frequency.value}"


@unique
class DataSector(Enum):
    sector_wics_big = "sector_wics_big"

    def as_sector(self,
                  exchange: str,
                  asset: str,
                  frequency: str) -> str:
        return f"{exchange.value}_{asset.value}_{self.value}_{frequency.value}"


@unique
class DataConstituents(Enum):
    const = "const"

    def as_const(self,
                 exchange: str,
                 asset: str,
                 frequency: str) -> str:
        return f"{exchange.value}_{asset.value}_{self.value}_{frequency.value}"


@unique
class DataName(Enum):
    name = "name"

    def as_name(self,
                exchange: str,
                asset: str,
                frequency: str) -> str:
        return f"{exchange.value}_{asset.value}_{self.value}_{frequency.value}"


@unique
class DataPool(Enum):
    def __new__(cls,
                data_class: Type[Enum],
                data_method: str,
                exchange: Market,
                asset: Asset,
                frequency: Frequency,
                source: Source):
        obj = object.__new__(cls)
        method: Callable[[Market, Asset, Frequency],
                         str] = getattr(data_class, data_method)
        base = method(exchange, asset, frequency)
        obj._value_ = f"{base}_{source.value}"
        return obj

    def __str__(self):
        return self._value_

    def __repr__(self):
        return self._value_

    # NOTE: PRICE DATA
    KOSPI_stock_price_o_1d = (DataPrice.price_o, "as_price",
                              Market.KOSPI, Asset.stock, Frequency.d, Source.quantiwise)
    KOSPI_stock_price_h_1d = (DataPrice.price_h, "as_price",
                              Market.KOSPI, Asset.stock, Frequency.d, Source.quantiwise)
    KOSPI_stock_price_l_1d = (DataPrice.price_l, "as_price",
                              Market.KOSPI, Asset.stock, Frequency.d, Source.quantiwise)
    KOSPI_stock_price_c_1d = (DataPrice.price_c, "as_price",
                              Market.KOSPI, Asset.stock, Frequency.d, Source.quantiwise)
    KOREA_stock_price_c_1d = (DataPrice.price_c, "as_price",
                              Market.KOREA, Asset.stock, Frequency.d, Source.quantiwise)

    KOSPI_stock_float_mktcap_1d = (
        DataPrice.float_mktcap, "as_price", Market.KOSPI, Asset.stock, Frequency.d, Source.quantiwise)

    # NOTE: VOLUME DATA
    KOSPI_stock_volume_1d = (DataVolume.volume, "as_volume",
                             Market.KOSPI, Asset.stock, Frequency.d, Source.quantiwise)

    # NOTE: SECTOR DATA
    KOREA_stock_sector_wics_big_n = (
        DataSector.sector_wics_big, "as_sector", Market.KOREA, Asset.stock, Frequency.n, Source.quantiwise)

    # NOTE: CONSTITUENT DATA
    KOSPI_stock_const_n = (
        DataConstituents.const, "as_const", Market.KOSPI, Asset.stock, Frequency.n, Source.dataguide
    )
    KOSPI200_stock_const_n = (
        DataConstituents.const, "as_const", Market.KOSPI200, Asset.stock, Frequency.n, Source.quantiwise
    )

    # NOTE: NAME DATA
    KOREA_stock_name_n = (DataName.name, "as_name", Market.KOREA,
                          Asset.stock, Frequency.n, Source.quantiwise)


class DataLoader:
    def __init__(self):
        pass

    def __call__(self, data_name: str) -> pd.DataFrame:
        member = DataPool.__members__.get(data_name)
        if member is None:
            raise ValueError(f"Invalid data name: {data_name}")

        file_path = os.path.join("./loader_data", f"{member.value}.pkl")
        return pd.read_pickle(file_path)


if __name__ == "__main__":
    loader = DataLoader()
    df = loader("KOSPI_stock_price_c_1d")
