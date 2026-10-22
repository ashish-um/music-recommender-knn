"""Results tab: evaluation of the current model (notebook sections 5.5 - 5.7)."""

import numpy as np
import pandas as pd
import seaborn as sns
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QRadioButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..widgets.dataframe_model import DataFrameModel, make_table_view
from ..widgets.mpl_canvas import PlotWidget

DEFAULT_COLOR = "#a8b5cf"
TUNED_COLOR = "#5570e8"


def _which_selector(on_change):
    """Classifier radio buttons, returning the selected results key."""
    row = QHBoxLayout()
    group = QButtonGroup(row)
    tuned = QRadioButton("Optimized KNN")
    default = QRadioButton("Default KNN")
    tuned.setChecked(True)
    tree = QRadioButton("Decision Tree")
    bayes = QRadioButton("Naïve Bayes")
    for button, key in ((tuned, "tuned"), (default, "default"),
                        (tree, "Decision Tree"), (bayes, "Naïve Bayes")):
        button.setProperty("results_key", key)
        group.addButton(button)
        row.addWidget(button)
    row.addStretch()
    group.buttonToggled.connect(lambda button, checked: checked and on_change())
    return row, lambda: group.checkedButton().property("results_key")


class ResultsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.model = None

        self.views = QListWidget()
        self.views.setObjectName("resultNavigation")
        self.views.setFixedWidth(180)
        self.views.setAccessibleName("Result views")
        self.stack = QStackedWidget()
        self.views.currentRowChanged.connect(self.stack.setCurrentIndex)

        self.header = QLabel()
        self.header.setWordWrap(True)
        self.header.setTextFormat(Qt.TextFormat.RichText)

        # 1. Summary table
        self.summary_model = DataFrameModel()
        self._add_view("Summary", make_table_view(self.summary_model))

        # 2. Comparison chart
        self.comparison_plot = PlotWidget()
        self._add_view("Before vs after chart", self.comparison_plot)

        self.classifier_model = DataFrameModel()
        self.classifier_plot = PlotWidget()
        self.classifier_note = QLabel()
        self.classifier_note.setWordWrap(True)
        classifier_page = QWidget()
        classifier_layout = QVBoxLayout(classifier_page)
        classifier_layout.addWidget(self.classifier_note)
        classifier_table = make_table_view(self.classifier_model)
        classifier_table.setMinimumHeight(170)
        classifier_layout.addWidget(classifier_table)
        classifier_layout.addWidget(self.classifier_plot, 1)
        self.classifier_view_index = self.stack.count()
        self._add_view("Model comparison", classifier_page)

        # 3. Confusion matrix
        self.cm_plot = PlotWidget()
        self.cm_plot.setMinimumHeight(400)
        selector, self.cm_which = _which_selector(self.draw_confusion_matrix)
        self._add_view("Confusion matrix", self._with_selector(selector, self.cm_plot))

        # 4. Per-genre report
        self.report_model = DataFrameModel(float_format="{:.2f}")
        self.report_plot = PlotWidget(toolbar=False)
        selector, self.report_which = _which_selector(self.show_report)
        report_split = QSplitter(Qt.Orientation.Vertical)
        report_split.addWidget(make_table_view(self.report_model, sortable=True))
        report_split.addWidget(self.report_plot)
        self._add_view("Genre-wise report", self._with_selector(selector, report_split))

        # 5. Accuracy vs K
        self.k_plot = PlotWidget()
        self._add_view("Accuracy vs K", self.k_plot)

        # 6. Cross validation
        self.cv_plot = PlotWidget()
        self._add_view("Cross validation", self.cv_plot)

        # 7. Grid search
        self.grid_model = DataFrameModel()
        self.grid_plot = PlotWidget()
        grid_split = QSplitter(Qt.Orientation.Vertical)
        grid_split.addWidget(make_table_view(self.grid_model, sortable=True))
        grid_split.addWidget(self.grid_plot)
        grid_split.setSizes([250, 350])
        self._add_view("Grid search", grid_split)

        right = QVBoxLayout()
        right.addWidget(self.header)
        right.addWidget(self.stack, 1)
        layout = QHBoxLayout(self)
        layout.addWidget(self.views)
        layout.addLayout(right, 1)
        self.views.setCurrentRow(0)

    def _add_view(self, name, widget):
        self.views.addItem(name)
        self.views.item(self.views.count() - 1).setSizeHint(QSize(160, 36))
        self.stack.addWidget(widget)

    @staticmethod
    def _with_selector(selector, widget):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(selector)
        layout.addWidget(widget, 1)
        return page

    def set_model(self, model):
        self.model = model
        if model is None:
            return
        params = model.best_params
        grid_note = "" if "grid" in model.results else " (grid search was not run)"
        self.header.setText(
            f"<b>Optimized parameters{grid_note}:</b> K = {params['n_neighbors']}, "
            f"weights = {params['weights']}, metric = {params['metric']} &nbsp;·&nbsp; "
            f"Train / test: {model.results['split']['train']:,} / {model.results['split']['test']:,} songs"
            f" &nbsp;·&nbsp; Trained {model.created_at.replace('T', ' ')}")
        self.show_summary()
        self.draw_comparison()
        self.show_classifier_comparison()
        self.draw_confusion_matrix()
        self.show_report()
        self.draw_k_sweep()
        self.draw_cv()
        self.show_grid()
        self.views.setCurrentRow(self.classifier_view_index if "model_comparison" in model.results else 0)

    # ---------- summary ----------

    def _comparison_rows(self):
        r = self.model.results
        k = self.model.config.precision_k
        rows = []
        for name in ("Accuracy", "Precision", "Recall", "F1-Score"):
            rows.append((name, r["default"]["metrics"][name], r["tuned"]["metrics"][name]))
        if "cv" in r and "grid" in r:
            rows.append(("CV Accuracy", r["cv"]["mean"], r["grid"]["best_score"]))
        rows.append((f"Precision@{k}", r["default"]["precision_at_k"], r["tuned"]["precision_at_k"]))
        return rows

    def show_summary(self):
        rows = self._comparison_rows()
        table = pd.DataFrame(rows, columns=["Metric", "Default KNN", "Optimized KNN"])
        table["Improvement"] = table["Optimized KNN"] - table["Default KNN"]
        table["Improvement"] = table["Improvement"].map(lambda v: f"{v:+.4f}")
        self.summary_model.set_dataframe(table)

    def draw_comparison(self):
        rows = self._comparison_rows()
        names = [r[0] for r in rows]
        x = np.arange(len(rows))
        self.comparison_plot.clear()
        ax = self.comparison_plot.figure.add_subplot()
        bars1 = ax.bar(x - 0.2, [r[1] for r in rows], width=0.4, label="Default KNN", color=DEFAULT_COLOR)
        bars2 = ax.bar(x + 0.2, [r[2] for r in rows], width=0.4, label="Optimized KNN", color=TUNED_COLOR)
        ax.bar_label(bars1, fmt="%.3f", fontsize=8)
        ax.bar_label(bars2, fmt="%.3f", fontsize=8)
        ax.set_xticks(x, names)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Default vs Optimized KNN")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        self.comparison_plot.draw()

    # ---------- confusion matrix and per-genre report ----------

    def show_classifier_comparison(self):
        comparison = self.model.results.get("model_comparison")
        self.classifier_plot.clear()
        if not comparison:
            self.classifier_model.set_dataframe(pd.DataFrame())
            self.classifier_note.setText("Retrain this model to compare KNN, Decision Tree and Naïve Bayes.")
            self.classifier_plot.message("Model comparison is unavailable in this saved model.")
            return
        table = pd.DataFrame([{"Model": name, **result["metrics"]}
                              for name, result in comparison.items()])
        table = table.sort_values("Accuracy", ascending=False, kind="stable").reset_index(drop=True)
        self.classifier_model.set_dataframe(table)
        best = table["Accuracy"].max()
        winners = table.loc[table["Accuracy"] == best, "Model"].tolist()
        knn_note = "grid-search optimized" if "grid" in self.model.results else "default settings"
        self.classifier_note.setText(
            f"Highest test accuracy: {', '.join(winners)} ({best:.2%}). "
            f"KNN: {knn_note}; Decision Tree and Naïve Bayes: default settings. "
            "Same features and split for all models. This compares genre prediction, not recommendation quality.")
        ax = self.classifier_plot.figure.add_subplot()
        bars = ax.bar(table["Model"], table["Accuracy"],
                      color=[TUNED_COLOR if a == best else DEFAULT_COLOR for a in table["Accuracy"]])
        ax.bar_label(bars, labels=[f"{a:.2%}" for a in table["Accuracy"]])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Test accuracy")
        ax.set_title("Genre classification: KNN vs Decision Tree vs Naïve Bayes")
        self.classifier_plot.draw()

    def _selected_result(self, which):
        if which in ("default", "tuned"):
            return self.model.results[which]
        return self.model.results.get("model_comparison", {}).get(which)

    def draw_confusion_matrix(self):
        if self.model is None:
            return
        which = self.cm_which()
        result = self._selected_result(which)
        if result is None:
            self.cm_plot.message("Retrain this model to see this classifier's results.")
            return
        cm = result["confusion_matrix"]
        names = self.model.genre_classes
        self.cm_plot.clear()
        ax = self.cm_plot.figure.add_subplot()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names,
                    ax=ax, annot_kws={"fontsize": 8 if len(names) <= 15 else 6})
        title = {"tuned": "Optimized KNN", "default": "Default KNN"}.get(which, which)
        ax.set_title(f"Confusion Matrix - {title}  ({np.trace(cm)} of {cm.sum()} correct)")
        ax.set_xlabel("Predicted Genre")
        ax.set_ylabel("Actual Genre")
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        self.cm_plot.draw()

    def show_report(self):
        if self.model is None:
            return
        result = self._selected_result(self.report_which())
        if result is None:
            self.report_model.set_dataframe(pd.DataFrame())
            self.report_plot.message("Retrain this model to see this classifier's results.")
            return
        report = result["per_class"]
        table = report.reset_index().rename(columns={"index": "genre"})
        self.report_model.set_dataframe(table)

        self.report_plot.clear()
        ax = self.report_plot.figure.add_subplot()
        ordered = report.sort_values("f1-score", ascending=False)
        ax.bar(ordered.index, ordered["f1-score"], color=TUNED_COLOR if self.report_which() == "tuned"
               else DEFAULT_COLOR)
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1-Score")
        ax.set_title("F1-Score by genre")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        self.report_plot.draw()

    # ---------- K sweep, CV, grid search ----------

    def draw_k_sweep(self):
        sweep = self.model.results.get("k_sweep")
        if not sweep:
            self.k_plot.message("Accuracy vs K was not run for this model.")
            return
        ks, accs = zip(*sweep)
        best = int(np.argmax(accs))
        self.k_plot.clear()
        ax = self.k_plot.figure.add_subplot()
        ax.plot(ks, accs, marker="o", color=DEFAULT_COLOR)
        ax.scatter([ks[best]], [accs[best]], s=150, facecolors="none", edgecolors=TUNED_COLOR, linewidths=2,
                   label=f"Best K = {ks[best]} (accuracy {accs[best]:.4f})", zorder=3)
        ax.set_title("Accuracy vs Number of Neighbors (K)")
        ax.set_xlabel("K")
        ax.set_ylabel("Testing Accuracy")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        self.k_plot.draw()

    def draw_cv(self):
        cv = self.model.results.get("cv")
        if not cv:
            self.cv_plot.message("Cross validation was not run for this model.")
            return
        scores = cv["scores"]
        folds = [f"Fold {i}" for i in range(1, len(scores) + 1)]
        self.cv_plot.clear()
        ax = self.cv_plot.figure.add_subplot()
        bars = ax.bar(folds, scores, color=DEFAULT_COLOR)
        ax.bar_label(bars, fmt="%.4f", fontsize=8)
        ax.axhline(cv["mean"], color=TUNED_COLOR, linestyle="--",
                   label=f"Mean = {cv['mean'] * 100:.2f} %   Std = {cv['std']:.4f}")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{len(scores)}-fold cross validation of the default KNN")
        ax.legend(loc="lower right")
        self.cv_plot.draw()

    def show_grid(self):
        grid = self.model.results.get("grid")
        if not grid:
            self.grid_model.set_dataframe(pd.DataFrame({"": ["Grid search was not run for this model."]}))
            self.grid_plot.message("Grid search was not run for this model.")
            return
        table = grid["table"]
        self.grid_model.set_dataframe(table)

        self.grid_plot.clear()
        ax = self.grid_plot.figure.add_subplot()
        for (metric, weights), group in table.groupby(["metric", "weights"]):
            group = group.sort_values("K")
            ax.plot(group["K"], group["mean CV accuracy"], marker="o",
                    linestyle="-" if weights == "distance" else "--", label=f"{metric}, {weights}")
        best = table.iloc[0]
        ax.scatter([best["K"]], [best["mean CV accuracy"]], s=160, facecolors="none", edgecolors="red",
                   linewidths=2, zorder=3)
        ax.set_xlabel("K")
        ax.set_ylabel("Mean CV accuracy")
        ax.set_title("Grid search: mean CV accuracy of every combination (best circled)")
        ax.legend(fontsize=8, ncols=2)
        ax.grid(True, linestyle="--", alpha=0.4)
        self.grid_plot.draw()
