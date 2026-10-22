"""Show a pandas DataFrame in a QTableView."""

import numpy as np
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtWidgets import QAbstractItemView, QHeaderView, QTableView


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None, parent=None, float_format="{:.4f}", headers=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()
        self.float_format = float_format
        self.headers = headers or {}       # column name -> header text
        self.row_brush = None              # callable(row Series) -> QColor | None

    def set_dataframe(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df.reset_index(drop=True)
        self.endResetModel()

    def dataframe(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._df.iat[index.row(), index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            if isinstance(value, (bool, np.bool_)):
                return "✓" if value else ""
            if isinstance(value, (float, np.floating)):
                return self.float_format.format(value)
            return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)):
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            if isinstance(value, (bool, np.bool_)):
                return Qt.AlignmentFlag.AlignCenter
        if role == Qt.ItemDataRole.BackgroundRole and self.row_brush:
            return self.row_brush(self._df.iloc[index.row()])
        if role == Qt.ItemDataRole.ToolTipRole:
            return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            column = str(self._df.columns[section])
            return self.headers.get(column, column)
        return str(section + 1)

    def sort(self, column, order=Qt.SortOrder.AscendingOrder):
        if self._df.empty:
            return
        self.layoutAboutToBeChanged.emit()
        self._df = self._df.sort_values(
            self._df.columns[column], ascending=order == Qt.SortOrder.AscendingOrder, kind="stable"
        ).reset_index(drop=True)
        self.layoutChanged.emit()


def make_table_view(model: DataFrameModel, sortable=False, stretch_columns=()) -> QTableView:
    """A read-only table. stretch_columns share the spare width; other columns fit their contents."""
    view = QTableView()
    view.setModel(model)
    view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    view.setAlternatingRowColors(model.row_brush is None)
    view.setShowGrid(False)
    view.verticalHeader().setDefaultSectionSize(34)
    view.verticalHeader().setVisible(False)
    view.setSortingEnabled(sortable)
    header = view.horizontalHeader()
    if not stretch_columns:
        header.setMaximumSectionSize(360)
    header.setStretchLastSection(not stretch_columns)

    # Columns may change after the view is built, so set the widths whenever the model resets.
    def fit_columns():
        if stretch_columns:
            for col in range(model.columnCount()):
                mode = QHeaderView.ResizeMode.Stretch if col in stretch_columns \
                    else QHeaderView.ResizeMode.ResizeToContents
                header.setSectionResizeMode(col, mode)
        else:
            view.resizeColumnsToContents()
    model.modelReset.connect(fit_columns)
    return view
