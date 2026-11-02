"""TV2 - Huy: Chuẩn hóa dữ liệu và trực quan hóa PCA 2D."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EXCEL_PATH = ROOT / "data/raw/D08 - wifi_localization.xlsx"
CSV_PATH = ROOT / "data/processed/wifi_localization.csv"
OUTPUT_DIR = ROOT / "figures/pca"

FEATURES = [f"WiFi_{i}" for i in range(1, 8)]
TARGET = "Room"
COLUMNS = FEATURES + [TARGET]
RANDOM_STATE = 42


def load_data():
    """Đọc đúng sheet, giữ mẫu đầu và không ghi đè CSV khác dữ liệu gốc."""
    raw = pd.read_excel(
        EXCEL_PATH,
        sheet_name="wifi_localization",
        header=None,
    )

    if raw.shape != (2000, 8):
        raise ValueError(f"Excel có kích thước {raw.shape}, cần kiểm tra với TV1.")

    raw.columns = COLUMNS
    raw = raw.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("Dữ liệu có giá trị thiếu hoặc vô hạn.")

    if not raw[TARGET].isin([1, 2, 3, 4]).all():
        raise ValueError("Nhãn phòng phải thuộc 1, 2, 3, 4.")

    raw[TARGET] = raw[TARGET].astype("int64")
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CSV_PATH.exists():
        existing = pd.read_csv(CSV_PATH)

        if list(existing.columns) != COLUMNS:
            raise ValueError("Header CSV khác quy ước. Cần đối chiếu với TV1.")

        existing = existing.apply(pd.to_numeric, errors="raise")

        if existing.shape != raw.shape or not np.array_equal(
            existing.to_numpy(dtype=float),
            raw.to_numpy(dtype=float),
        ):
            raise ValueError(
                "CSV khác Excel về dữ liệu hoặc thứ tự dòng. "
                "Không ghi đè; cần thống nhất với TV1."
            )
    else:
        raw.to_csv(CSV_PATH, index=False, encoding="utf-8")

    # Dùng CSV làm đầu vào phân tích theo yêu cầu đề.
    return pd.read_csv(CSV_PATH)


def save_figure(fig, filename):
    """Xuất cả PNG để chèn báo cáo và PDF để giữ chất lượng hình."""
    for extension in ("png", "pdf"):
        fig.savefig(
            OUTPUT_DIR / f"{filename}.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

    df = load_data()
    X = df[FEATURES].astype("float64")
    y = df[TARGET].astype("int64")

    # Nhãn Room không tham gia bước chuẩn hóa hoặc PCA.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(
            n_components=2,
            svd_solver="full",
            random_state=RANDOM_STATE,
        )),
    ])

    coordinates = pipeline.fit_transform(X)
    scaler = pipeline.named_steps["scaler"]
    pca = pipeline.named_steps["pca"]
    scaled = scaler.transform(X)
    ratios = pca.explained_variance_ratio_

    # Kiểm tra kết quả chuẩn hóa và kích thước đầu ra.
    assert coordinates.shape == (len(df), 2)
    assert np.isfinite(coordinates).all()
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-10)

    statistics = pd.DataFrame({
        "mean_raw": X.mean(),
        "std_raw": X.std(ddof=0),
        "variance_raw": X.var(ddof=0),
        "mean_scaled": scaled.mean(axis=0),
        "std_scaled": scaled.std(axis=0),
    })
    statistics.to_csv(
        OUTPUT_DIR / "pca_scaling_statistics.csv",
        index_label="feature",
    )

    variance = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": ratios,
        "cumulative_ratio": np.cumsum(ratios),
    })
    variance.to_csv(OUTPUT_DIR / "pca_variance.csv", index=False)

    # sample_id đánh số theo thứ tự mẫu trong CSV, bắt đầu từ 1.
    points = pd.DataFrame({
        "sample_id": np.arange(1, len(df) + 1),
        "PC1": coordinates[:, 0],
        "PC2": coordinates[:, 1],
        "Room": y.to_numpy(),
    })
    points.to_csv(OUTPUT_DIR / "pca_coordinates.csv", index=False)

    # Biểu đồ tỷ lệ phương sai giữ lại.
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    bars = ax.bar(
        ["PC1", "PC2"],
        ratios * 100,
        color=["#0072B2", "#009E73"],
    )
    ax.bar_label(bars, labels=[f"{r:.2%}" for r in ratios], padding=5)
    ax.set_ylim(0, max(ratios) * 100 + 12)
    ax.set_ylabel("Phương sai giải thích (%)")
    ax.set_title(f"PCA — Tổng phương sai giữ lại: {ratios.sum():.2%}")
    save_figure(fig, "pca_explained_variance")

    # Scatter plot: mỗi phòng có màu và marker riêng.
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    markers = ["o", "s", "^", "D"]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for room, color, marker in zip(range(1, 5), colors, markers):
        mask = y.to_numpy() == room
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=color,
            marker=marker,
            s=23,
            alpha=0.55,
            linewidths=0,
            label=f"Phòng {room} (n={int(mask.sum())})",
        )

    ax.set_xlabel(f"PC1 ({ratios[0]:.2%} phương sai)")
    ax.set_ylabel(f"PC2 ({ratios[1]:.2%} phương sai)")
    ax.set_title("WiFi Localization — StandardScaler + PCA 2D")
    ax.grid(alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(title="Nhãn thực tế")
    save_figure(fig, "pca_rooms_2d")

    print(f"Dữ liệu: {len(df)} mẫu, {X.shape[1]} đặc trưng")
    print(f"PC1: {ratios[0]:.4%}")
    print(f"PC2: {ratios[1]:.4%}")
    print(f"Tổng phương sai giữ lại: {ratios.sum():.4%}")
    print("\nThống kê chuẩn hóa:")
    print(statistics.round(6).to_string())

    for feature in ("WiFi_1", "WiFi_4"):
        ratio = statistics.loc[feature, "std_raw"] / statistics.loc["WiFi_2", "std_raw"]
        print(
            f"\n{feature}/WiFi_2: "
            f"tỷ số độ lệch chuẩn = {ratio:.4f}; "
            f"tỷ số phương sai = {ratio ** 2:.4f}"
        )

    print(f"\nHoàn thành. Kết quả lưu tại: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()