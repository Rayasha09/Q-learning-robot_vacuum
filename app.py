from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

from train import GRID_SIZE, LEVEL, LEVELS


BASE_DIR = Path(__file__).resolve().parent
GIF_PATH = BASE_DIR / "robot_vacuum.gif"
CURVE_PATH = BASE_DIR / "learning_curve.png"

CURRENT_LEVEL = LEVELS.get(LEVEL, LEVELS["medium"])
OBSTACLES = CURRENT_LEVEL["obstacles"]
DIRT_SPOTS = CURRENT_LEVEL["dirt_spots"]
START = CURRENT_LEVEL["start"]
REWARDS = CURRENT_LEVEL["rewards"]


def draw_apartment_map():
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(GRID_SIZE + 1))
    ax.set_yticks(range(GRID_SIZE + 1))
    ax.grid(True, color="#d0d7de", linewidth=1)
    ax.set_facecolor("#f8fafc")

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            rect = patches.Rectangle((col, row), 1, 1, facecolor="#ffffff", edgecolor="none")
            ax.add_patch(rect)

    for row, col in OBSTACLES:
        rect = patches.FancyBboxPatch(
            (col + 0.08, row + 0.08), 0.84, 0.84,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#d65c5c", edgecolor="#ffffff", linewidth=2
        )
        ax.add_patch(rect)

    for row, col in DIRT_SPOTS:
        rect = patches.FancyBboxPatch(
            (col + 0.08, row + 0.08), 0.84, 0.84,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="#f0cf72", edgecolor="#ffffff", linewidth=2
        )
        ax.add_patch(rect)
        dot = patches.Circle((col + 0.5, row + 0.5), 0.09, color="#5b6168")
        ax.add_patch(dot)

    robot = patches.Circle((START[1] + 0.5, START[0] + 0.5), 0.26, color="#3f83f8")
    ax.add_patch(robot)
    eye1 = patches.Circle((START[1] + 0.42, START[0] + 0.43), 0.03, color="#1f2937")
    eye2 = patches.Circle((START[1] + 0.58, START[0] + 0.43), 0.03, color="#1f2937")
    ax.add_patch(eye1)
    ax.add_patch(eye2)

    ax.set_title("Карта квартиры", fontsize=16, pad=14)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return fig


st.set_page_config(page_title="Робот-пылесос", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
        text-align: center;
    }
    .section-box {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 18px;
        padding: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Робот-пылесос с Q-learning")
st.write(
    "Этот проект показывает, как робот-пылесос обучается убирать квартиру с помощью "
    "Q-learning. Робот получает награды за уборку мусора и штрафы за ошибки."
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div class='metric-card'><h4>Уровень</h4><h2>{LEVEL}</h2></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><h4>Размер сетки</h4><h2>{GRID_SIZE}x{GRID_SIZE}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h4>Мусор</h4><h2>{len(DIRT_SPOTS)}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='metric-card'><h4>Препятствия</h4><h2>{len(OBSTACLES)}</h2></div>", unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("Сама игра")
    st.pyplot(draw_apartment_map(), use_container_width=True)
    st.caption("Синий робот стартует из фиксированной точки и учится убирать жёлтые зоны.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.subheader("Награды и наказания")
    st.write(f"За уборку новой зоны: `{REWARDS['clean']}`")
    st.write(f"За завершение всей уборки: `{REWARDS['finish']}`")
    st.write(f"За обычный шаг: `{REWARDS['step']}`")
    st.write(f"За удар о стену: `{REWARDS['wall']}`")
    st.write(f"За удар о мебель: `{REWARDS['obstacle']}`")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("GIF работы обученного робота")
if GIF_PATH.exists():
    with open(GIF_PATH, "rb") as gif_file:
        st.image(gif_file.read(), caption="Робот убирает квартиру после обучения", use_container_width=True)
else:
    st.warning("Файл robot_vacuum.gif не найден. Сначала запустите train.py")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("График обучения")
if CURVE_PATH.exists():
    st.image(str(CURVE_PATH), caption="Learning curve", use_container_width=True)
else:
    st.warning("Файл learning_curve.png не найден. Сначала запустите train.py")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='section-box'>", unsafe_allow_html=True)
st.subheader("Как запустить")
st.code("python3 train.py\nstreamlit run app.py", language="bash")
st.markdown("</div>", unsafe_allow_html=True)
