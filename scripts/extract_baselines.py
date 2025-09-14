from casacore.tables import table, tablecolumn
from dataclasses import dataclass
from typing import Tuple, Union, List
import numpy as np
import pandas as pd

@dataclass
class Baseline:
    # The (ANTENNA1, ANTENNA2) pair for the baseline
    antenna: Tuple[int, int]
    # The first row of the baseline
    start: int
    # The stride for the baseline
    stride: int
    # The number of data points in the baseline
    count: int

    def get_data(self, col: tablecolumn) -> np.ndarray:
        """Retrieves column data for the baseline"""
        return col.getcol(startrow=self.start, nrow=self.count, rowincr=self.stride)

    def set_data(self, col: tablecolumn, data: np.ndarray):
        "Writes data for a baseline back into a column"
        return col.putcol(data, startrow=self.start, nrow=self.count, rowincr=self.stride)

    def to_dict(self) -> dict:
        return {
            "a1": self.antenna[0],
            "a2": self.antenna[1],
            "start": self.start,
            "stride": self.stride,
            "count": self.count,
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog="Baseline Extraction",
        description="Determines all the baselines for a measurement set and writes it to a CSV"
    )
    parser.add_argument("ms", help="Path to the measurement set")
    parser.add_argument("-o", "--output", required=True, help="Output of the baseline CSV")

    args = parser.parse_args()


    ms_path = args.ms 
    csv_path = args.output 


    print(f"Loading measurement set '{ms_path}'")
    ms = table(ms_path)
    antenna1 = ms.getcol("ANTENNA1")
    antenna2 = ms.getcol("ANTENNA2")

    combinations = set(zip(antenna1, antenna2))
    baselines: List[Baseline] = []

    print("Computing baselines")
    for a1, a2 in combinations:
        # Get all the data indices corresponding to the antenna pair
        indices = np.argwhere((antenna1==a1)&(antenna2==a2)).T[0]
        if len(indices) < 1:
            continue

        # Compute the stride between each data point
        strides = set(np.diff(indices))
        assert len(strides) == 1, f"Stride is not constant in baseline ({a1}, {a2}): {strides}"

        # Add a new baseline
        baselines.append(Baseline(
            antenna=(a1,a2),
            start=indices[0],
            stride=next(iter(strides)),
            count=len(indices)
        ))

    df = pd.DataFrame.from_records([b.to_dict() for b in baselines])
    df = df.sort_values(["a1", "a2"])
    df.to_csv(csv_path, index=False)
    print(f"Baselines written to '{csv_path}'")
